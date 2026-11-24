"""Media sanitization (PLAN 14.8): strip EXIF/GPS from images, validate videos with ffprobe.

Images are decoded once, EXIF orientation is applied to the pixels, and the pixels are re-saved
into a fresh image so no metadata (EXIF, GPS, XMP, text chunks) survives. Videos are probed
before any decode against a codec allow-list and size/resolution/fps/duration/frame limits.
Failures raise ``MediaError`` with a typed, permanent code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from countonce.errors import ErrorCode, MediaError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class MediaLimits:
    max_image_bytes: int = 50 * 1024 * 1024
    max_video_bytes: int = 4 * 1024**3
    max_pixels: int = 100_000_000  # decompression-bomb guard, checked before full decode
    max_video_width: int = 7680
    max_video_height: int = 4320
    max_fps: float = 120.0
    max_duration_s: float = 600.0
    max_video_frames: int = 36_000
    codecs: tuple[str, ...] = ("h264", "hevc", "vp8", "vp9", "av1", "mpeg4")
    ffprobe_timeout_s: float = 30.0


DEFAULT_LIMITS = MediaLimits()


@dataclass
class SanitizeReport:
    """What sanitization did; ``exif_stripped`` is recorded in result provenance."""

    images: int = 0
    exif_stripped: bool = False  # True once at least one image carried (and lost) metadata
    images_with_metadata: int = 0
    videos_validated: int = 0
    gps_found: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "images_sanitized": self.images,
            "images_with_metadata": self.images_with_metadata,
            "gps_found": self.gps_found,
            "videos_validated": self.videos_validated,
            "exif_stripped": self.exif_stripped or self.images > 0,
        }


def _reject(code: ErrorCode, message: str) -> MediaError:
    return MediaError(message, code=code)


def check_extension(path: Path) -> str:
    """Return "image" or "video"; raise ERR_UNSUPPORTED_MEDIA otherwise."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise _reject(
        ErrorCode.ERR_UNSUPPORTED_MEDIA, f"unsupported file type {ext or '(none)'}: {path.name}"
    )


_GPS_IFD = 0x8825


def sanitize_image(
    src: Path,
    out_dir: Path,
    limits: MediaLimits = DEFAULT_LIMITS,
    report: SanitizeReport | None = None,
) -> Path:
    """Write a metadata-free, orientation-applied copy of ``src`` into ``out_dir``."""
    src = Path(src)
    if check_extension(src) != "image":
        raise _reject(ErrorCode.ERR_UNSUPPORTED_MEDIA, f"not an image: {src.name}")
    if src.stat().st_size > limits.max_image_bytes:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{src.name} exceeds {limits.max_image_bytes} bytes"
        )
    try:
        with Image.open(src) as img:
            w, h = img.size  # header only; check before the full decode
            if w * h > limits.max_pixels:
                raise _reject(
                    ErrorCode.ERR_FILE_TOO_LARGE, f"{src.name} is {w}x{h}, over the pixel limit"
                )
            exif = img.getexif()
            had_meta = bool(exif) or bool(img.info.get("exif") or img.info.get("xmp"))
            has_gps = bool(exif.get_ifd(_GPS_IFD)) if exif else False
            img.load()  # raises on truncated data
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L", "RGBA"):
                img = img.convert(
                    "RGBA" if "transparency" in img.info or img.mode == "PA" else "RGB"
                )
            clean = Image.frombytes(img.mode, img.size, img.tobytes())  # fresh image: no .info
    except MediaError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise _reject(ErrorCode.ERR_CORRUPT_MEDIA, f"cannot decode {src.name}: {e}") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    if dst.resolve() == src.resolve():
        raise ValueError("sanitize output must not overwrite the source")
    suffix = src.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        if clean.mode == "RGBA":
            clean = clean.convert("RGB")
        clean.save(dst, "JPEG", quality=95, exif=b"")
    else:
        clean.save(dst)
    if report is not None:
        report.images += 1
        report.images_with_metadata += int(had_meta)
        report.gps_found += int(has_gps)
        report.exif_stripped = True
    return dst


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def validate_video(
    path: Path, limits: MediaLimits = DEFAULT_LIMITS, report: SanitizeReport | None = None
) -> dict[str, object]:
    """Probe ``path`` with ffprobe and enforce the limits. Returns the video stream facts."""
    path = Path(path)
    if check_extension(path) != "video":
        raise _reject(ErrorCode.ERR_UNSUPPORTED_MEDIA, f"not a video: {path.name}")
    if path.stat().st_size > limits.max_video_bytes:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{path.name} exceeds {limits.max_video_bytes} bytes"
        )
    if not ffprobe_available():
        raise RuntimeError("ffprobe not found on PATH; install ffmpeg to ingest video")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
        str(path),
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=limits.ffprobe_timeout_s)
    except subprocess.TimeoutExpired as e:
        raise _reject(ErrorCode.ERR_CORRUPT_MEDIA, f"ffprobe timed out on {path.name}") from e
    if proc.returncode != 0:
        raise _reject(
            ErrorCode.ERR_CORRUPT_MEDIA, f"ffprobe rejected {path.name}: {proc.stderr.strip()}"
        )
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise _reject(
            ErrorCode.ERR_CORRUPT_MEDIA, f"unreadable ffprobe output for {path.name}"
        ) from e
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise _reject(ErrorCode.ERR_CORRUPT_MEDIA, f"no video stream in {path.name}")
    s = streams[0]
    codec = str(s.get("codec_name", "")).lower()
    if codec not in limits.codecs:
        raise _reject(ErrorCode.ERR_UNSUPPORTED_CODEC, f"codec {codec or '?'} is not allowed")
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    if w <= 0 or h <= 0:
        raise _reject(ErrorCode.ERR_CORRUPT_MEDIA, f"{path.name} has no frame size")
    if w > limits.max_video_width or h > limits.max_video_height:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{path.name} is {w}x{h}, over the resolution limit"
        )
    fps = _parse_rate(s.get("avg_frame_rate") or s.get("r_frame_rate"))
    if fps > limits.max_fps:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{path.name} runs at {fps:g} fps, over the limit"
        )
    duration = _to_float(s.get("duration")) or _to_float(info.get("format", {}).get("duration"))
    if duration > limits.max_duration_s:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{path.name} is {duration:.0f}s, over the duration limit"
        )
    n_frames = int(_to_float(s.get("nb_frames")) or duration * fps)
    if n_frames > limits.max_video_frames:
        raise _reject(
            ErrorCode.ERR_FILE_TOO_LARGE, f"{path.name} has {n_frames} frames, over the limit"
        )
    if report is not None:
        report.videos_validated += 1
    return {
        "codec": codec,
        "width": w,
        "height": h,
        "fps": fps,
        "duration_s": duration,
        "frames": n_frames,
    }


def _to_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _parse_rate(v: object) -> float:
    try:
        num, _, den = str(v).partition("/")
        return float(num) / float(den) if den and float(den) else float(num)
    except (TypeError, ValueError):
        return 0.0
