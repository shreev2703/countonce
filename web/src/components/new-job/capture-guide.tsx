/** One-screen capture guide (the short form of docs/CAPTURE_GUIDE.md). */
export function CaptureGuide() {
  return (
    <details className="rounded-lg border border-border bg-card p-4" data-testid="capture-guide" open>
      <summary className="cursor-pointer text-base font-semibold">How to take the photos</summary>
      <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm">
        <li>Put the objects in one place and keep them still. Good, even light; no flash.</li>
        <li>Walk a slow arc around them. Take about 20 photos, each overlapping the last by roughly a third.</li>
        <li>Keep every object visible in at least two photos. Change your height a little.</li>
        <li>Do not zoom or crop. Hold the phone steady so photos are sharp.</li>
      </ol>
      <p className="mt-3 text-sm text-muted-foreground">
        The count is of objects visible in at least one photo. Objects hidden in every photo are not counted.
      </p>
    </details>
  );
}
