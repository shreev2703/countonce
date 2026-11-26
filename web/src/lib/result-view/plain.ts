/**
 * Plain-language wording for the results page. The raw phrasing (`intervalText`, warning messages,
 * model versions) stays available in the "Technical details" drawer.
 */
import type { Result } from "@/lib/schemas/result";

export function plainInterval(result: Result): string {
  const { low, high, method } = result.interval;
  return method === "sensitivity" ? `likely ${low}–${high} when settings are varied` : `likely ${low}–${high}`;
}

export function plainWarning(w: Result["warnings"][number], result: Result): string {
  if (w.code === "low_depth_confidence") return "The 3D reconstruction of this scene is uncertain.";
  if (w.code === "wide_range") {
    const { low, high } = result.interval;
    return `The count depends on the settings: it ranges from ${low} to ${high}.`;
  }
  return w.message;
}

/** True when the result came from the test backends (colour detector, known cameras), not real models. */
export function usesTestBackends(result: Result): boolean {
  const { detector, backbone } = result.model_versions;
  return detector.startsWith("color_blob") || backbone.startsWith("fixture");
}
