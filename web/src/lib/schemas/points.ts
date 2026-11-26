/** `points.bin` (CPC1) decoder; zod-free so the client viewer chunk stays small. */
export const POINTS_FORMAT_VERSION = 1;
export const POINTS_HEADER_BYTES = 12;

/** Decode `points.bin` (CPC1): header, float32 xyz (n,3), uint8 rgb (n,3), little endian. */
export function parsePoints(buf: ArrayBuffer): { xyz: Float32Array; rgb: Uint8Array } {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  const version = view.getUint32(4, true);
  const n = view.getUint32(8, true);
  if (magic !== "CPC1" || version !== POINTS_FORMAT_VERSION) {
    throw new Error(`unsupported points file ${magic} v${version}`);
  }
  if (buf.byteLength !== POINTS_HEADER_BYTES + n * 15) throw new Error("points file has the wrong length");
  const xyz = new Float32Array(buf, POINTS_HEADER_BYTES, n * 3);
  const rgb = new Uint8Array(buf, POINTS_HEADER_BYTES + n * 12, n * 3);
  return { xyz, rgb };
}
