/**
 * Browser pre-flight checks that run before anything is uploaded (PLAN 17.5): cheap, local, and
 * phrased as what to do. The limits mirror the backend (`countonce/ingest/sanitize.py`); the
 * server still re-checks everything, this only saves a failed upload.
 */
export const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
export const VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
export const MAX_IMAGE_BYTES = 50 * 1024 * 1024;
export const MAX_VIDEO_BYTES = 4 * 1024 ** 3;
export const MIN_PHOTOS = 6; // docs/CAPTURE_GUIDE.md: 20 recommended, 6-10 minimum
export const RECOMMENDED_PHOTOS = 20;
export const MIN_SHORT_SIDE_PX = 640;

export type Level = "pass" | "warn" | "fail";
export interface Check {
  id: string;
  level: Level;
  title: string;
  detail: string;
}

export interface FileInfo {
  name: string;
  size: number;
  /** Decoded pixel size, when the browser could decode it. */
  width?: number;
  height?: number;
}

export interface Env {
  online: boolean;
  hasCrypto: boolean;
  hasSlice: boolean;
}

export const kindOf = (name: string): "image" | "video" | null => {
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  if (IMAGE_EXTS.includes(ext)) return "image";
  if (VIDEO_EXTS.includes(ext)) return "video";
  return null;
};

const mb = (n: number) => `${(n / 1024 / 1024).toFixed(0)} MB`;

export function runPreflight(files: FileInfo[], env: Env): Check[] {
  const out: Check[] = [];
  const add = (id: string, level: Level, title: string, detail: string) => out.push({ id, level, title, detail });

  const browserOk = env.hasCrypto && env.hasSlice;
  add(
    "browser",
    browserOk ? "pass" : "fail",
    browserOk ? "Browser can upload" : "This browser cannot upload large files",
    browserOk ? "Resumable upload is available." : "Open this page over HTTPS in a current Chrome, Safari or Firefox.",
  );
  add(
    "network",
    env.online ? "pass" : "warn",
    env.online ? "You are online" : "You look offline",
    env.online ? "Uploads can start." : "Uploads resume when the connection is back; nothing is lost.",
  );

  if (files.length === 0) {
    add("count", "fail", "Add photos or a video", "Pick at least 6 photos, or one video that walks around the objects.");
    return out;
  }
  const bad = files.filter((f) => kindOf(f.name) === null);
  add(
    "types",
    bad.length ? "fail" : "pass",
    bad.length ? `${bad.length} file(s) are not photos or videos we can read` : "All files are photos or videos",
    bad.length
      ? `Remove ${bad.map((f) => f.name).slice(0, 3).join(", ")}${bad.length > 3 ? ", …" : ""}. Use JPG, PNG or WebP photos, or MP4/MOV video. iPhone HEIC is not supported yet: set the camera to Most Compatible.`
      : "Types are supported.",
  );
  const big = files.filter((f) => {
    const k = kindOf(f.name);
    return (k === "image" && f.size > MAX_IMAGE_BYTES) || (k === "video" && f.size > MAX_VIDEO_BYTES);
  });
  add(
    "size",
    big.length ? "fail" : "pass",
    big.length ? "A file is too large" : "File sizes are fine",
    big.length
      ? `${big[0]!.name} is ${mb(big[0]!.size)}. Photos can be up to ${mb(MAX_IMAGE_BYTES)}, videos up to 4 GB.`
      : "Under the limits.",
  );

  const images = files.filter((f) => kindOf(f.name) === "image");
  const videos = files.filter((f) => kindOf(f.name) === "video");
  if (videos.length === 0 && images.length < MIN_PHOTOS) {
    add("count", "fail", `Only ${images.length} photo(s)`, `Add at least ${MIN_PHOTOS}; ${RECOMMENDED_PHOTOS} walking slowly around the objects works best.`);
  } else if (videos.length === 0 && images.length < RECOMMENDED_PHOTOS) {
    add("count", "warn", `${images.length} photos`, `Enough to try. ${RECOMMENDED_PHOTOS} give a tighter count.`);
  } else {
    add("count", "pass", videos.length ? `${videos.length} video(s), ${images.length} photo(s)` : `${images.length} photos`, "Enough views.");
  }

  const small = images.filter((f) => f.width !== undefined && f.height !== undefined && Math.min(f.width, f.height) < MIN_SHORT_SIDE_PX);
  const unreadable = images.filter((f) => f.width === 0);
  if (unreadable.length) {
    add("decode", "fail", `${unreadable.length} photo(s) cannot be opened`, `${unreadable[0]!.name} looks corrupt. Re-export it or retake it.`);
  } else if (small.length) {
    add("resolution", "warn", `${small.length} photo(s) are small`, `Shorter side under ${MIN_SHORT_SIDE_PX} px (${small[0]!.name}). Small objects may be missed.`);
  } else if (images.length) {
    add("resolution", "pass", "Photos are sharp enough", "Resolution is fine.");
  }

  const seen = new Map<string, number>();
  for (const f of files) {
    const k = `${f.name}:${f.size}`;
    seen.set(k, (seen.get(k) ?? 0) + 1);
  }
  const dups = [...seen.values()].filter((n) => n > 1).length;
  if (dups) add("duplicates", "warn", "Some files were added twice", "Duplicate photos add no new viewpoint; remove them.");
  return out;
}

export const canSubmit = (checks: Check[]) => !checks.some((c) => c.level === "fail");

/** Decode a photo in the browser just to read its size; width 0 means it would not decode. */
export async function probeImage(file: File): Promise<FileInfo> {
  const info: FileInfo = { name: file.name, size: file.size };
  if (kindOf(file.name) !== "image") return info;
  try {
    const bmp = await createImageBitmap(file);
    info.width = bmp.width;
    info.height = bmp.height;
    bmp.close();
  } catch {
    info.width = 0;
    info.height = 0;
  }
  return info;
}
