"""Media sanitization and typed error taxonomy tests (PLAN 14.4, 14.8, 18)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from countonce.errors import (
    MAX_ATTEMPTS,
    ErrorCode,
    GpuOomError,
    MediaError,
    is_retryable,
    retry_decision,
    user_tip,
)
from countonce.ingest.sanitize import (
    MediaLimits,
    SanitizeReport,
    sanitize_image,
    validate_video,
)
from countonce.pipeline import collect_paths


def _gps_jpeg(path: Path) -> None:
    img = Image.new("RGB", (64, 48), (200, 30, 30))
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    exif[0x0112] = 1
    exif[0x8825] = {1: "N", 2: (35.0, 41.0, 0.0), 3: "E", 4: (139.0, 41.0, 0.0)}
    img.save(path, "JPEG", exif=exif)


def test_gps_exif_is_stripped(tmp_path):
    src = tmp_path / "a.jpg"
    _gps_jpeg(src)
    with Image.open(src) as im:
        assert im.getexif().get_ifd(0x8825)  # fixture really carries GPS
    rep = SanitizeReport()
    dst = sanitize_image(src, tmp_path / "out", report=rep)
    with Image.open(dst) as im:
        assert len(im.getexif()) == 0
        assert "exif" not in im.info
        assert im.size == (64, 48)
    assert b"TestCam" not in dst.read_bytes()
    assert rep.exif_stripped and rep.gps_found == 1 and rep.images_with_metadata == 1


def test_orientation_applied_to_pixels(tmp_path):
    src = tmp_path / "r.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    Image.new("RGB", (60, 40)).save(src, "JPEG", exif=exif)
    dst = sanitize_image(src, tmp_path / "out")
    with Image.open(dst) as im:
        assert im.size == (40, 60)


def test_truncated_jpeg_is_corrupt(tmp_path):
    src = tmp_path / "t.jpg"
    Image.effect_noise((256, 256), 80).convert("RGB").save(src, "JPEG")
    data = src.read_bytes()
    src.write_bytes(data[: len(data) // 2])
    with pytest.raises(MediaError) as ei:
        sanitize_image(src, tmp_path / "out")
    assert ei.value.code == ErrorCode.ERR_CORRUPT_MEDIA and not ei.value.retryable


def test_txt_is_unsupported(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("hello")
    with pytest.raises(MediaError) as ei:
        collect_paths(src, tmp_path / "o", 2.0)
    assert ei.value.code == ErrorCode.ERR_UNSUPPORTED_MEDIA


def test_oversized_file_and_pixels(tmp_path):
    src = tmp_path / "big.jpg"
    _gps_jpeg(src)
    with pytest.raises(MediaError) as ei:
        sanitize_image(src, tmp_path / "o", MediaLimits(max_image_bytes=10))
    assert ei.value.code == ErrorCode.ERR_FILE_TOO_LARGE
    with pytest.raises(MediaError) as ei2:
        sanitize_image(src, tmp_path / "o", MediaLimits(max_pixels=100))
    assert ei2.value.code == ErrorCode.ERR_FILE_TOO_LARGE


def test_collect_paths_sanitizes_and_reports(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    _gps_jpeg(d / "a.jpg")
    rep = SanitizeReport()
    paths = collect_paths(d, tmp_path / "o", 2.0, rep)
    assert paths and all("sanitized" in str(p) for p in paths)
    assert rep.to_dict()["exif_stripped"] is True
    with Image.open(paths[0]) as im:
        assert len(im.getexif()) == 0


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None, reason="ffmpeg missing"
)


def _make_video(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


@needs_ffmpeg
def test_video_validation(tmp_path):
    v = tmp_path / "v.mp4"
    try:
        _make_video(v)
    except subprocess.CalledProcessError:
        pytest.skip("no libx264")
    facts = validate_video(v)
    assert facts["codec"] == "h264" and facts["width"] == 64
    with pytest.raises(MediaError) as ei:
        validate_video(v, MediaLimits(codecs=("vp9",)))
    assert ei.value.code == ErrorCode.ERR_UNSUPPORTED_CODEC
    with pytest.raises(MediaError) as ei2:
        validate_video(v, MediaLimits(max_duration_s=0.1))
    assert ei2.value.code == ErrorCode.ERR_FILE_TOO_LARGE
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(MediaError) as ei3:
        validate_video(bad)
    assert ei3.value.code == ErrorCode.ERR_CORRUPT_MEDIA


def test_missing_ffprobe_is_clear_error(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr("countonce.ingest.sanitize.shutil.which", lambda _n: None)
    with pytest.raises(RuntimeError, match="ffprobe"):
        validate_video(v)


# --- taxonomy / retry policy ------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.ERR_CORRUPT_MEDIA,
        ErrorCode.ERR_UNSUPPORTED_MEDIA,
        ErrorCode.ERR_UNSUPPORTED_CODEC,
        ErrorCode.ERR_FILE_TOO_LARGE,
        ErrorCode.ERR_TOO_FEW_FRAMES,
        ErrorCode.ERR_NO_RECON,
        ErrorCode.ERR_NO_OVERLAP,
    ],
)
def test_permanent_errors_dead_letter_immediately(code):
    assert not is_retryable(code) and user_tip(code)
    assert retry_decision(code, 1).action == "dead_letter"


def test_retryable_backoff_is_exponential_then_capped_by_attempts():
    delays = []
    for attempt in range(1, MAX_ATTEMPTS):
        d = retry_decision(ErrorCode.ERR_STORAGE_TRANSIENT, attempt)
        assert d.retry
        delays.append(d.backoff_s)
    assert delays == [2.0 * 2**i for i in range(MAX_ATTEMPTS - 1)]
    assert retry_decision(ErrorCode.ERR_STORAGE_TRANSIENT, MAX_ATTEMPTS).action == "dead_letter"


def test_backoff_cap():
    d = retry_decision("ERR_WORKER_TIMEOUT", 5, max_attempts=10, base_s=10, cap_s=30)
    assert d.retry and d.backoff_s == 30


def test_gpu_oom_error_compat():
    from countonce.oom import ERR_GPU_OOM, USER_TIP
    from countonce.oom import GpuOomError as Reexported

    assert Reexported is GpuOomError and ERR_GPU_OOM == "ERR_GPU_OOM"
    e = GpuOomError("boom", attempts=["a"])
    assert e.code == ErrorCode.ERR_GPU_OOM and e.retryable and e.tip == USER_TIP
    assert str(e).startswith("ERR_GPU_OOM:") and e.attempts == ["a"]
    assert retry_decision(e, 1).retry
