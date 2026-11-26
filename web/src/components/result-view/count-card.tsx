"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { DemoScene } from "@/lib/result-view/demo";
import { confidenceBadge, csvForResult } from "@/lib/result-view/model";
import { plainInterval, usesTestBackends } from "@/lib/result-view/plain";
import { cn } from "@/lib/utils";

const BADGE = {
  High: "border-solid text-status-new",
  Medium: "border-dashed text-status-duplicate",
  Low: "border-dotted text-status-rejected",
} as const;

export function CountCard({ scene }: { scene: DemoScene }) {
  const { result } = scene;
  const conf = confidenceBadge(result);
  const [shared, setShared] = useState(false);

  const downloadCsv = () => {
    const url = URL.createObjectURL(new Blob([csvForResult(result)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${scene.manifest.scene_id}-count.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShared(true);
    } catch {
      setShared(false);
    }
  };

  return (
    <section aria-labelledby="count-heading" className="surface flex flex-col gap-4 rounded-2xl p-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h2 id="count-heading" className="sr-only">
          Count result
        </h2>
        <p className="text-sm text-muted-foreground">Counting &ldquo;{result.prompt}&rdquo;</p>
        <p
          data-testid="count"
          className="text-6xl font-semibold leading-none tabular-nums"
          aria-label={`Count: ${result.count} ${result.prompt}, ${plainInterval(result)}`}
        >
          {result.count}
        </p>
        <p className="mt-2 text-sm" data-testid="interval">
          {plainInterval(result)}
        </p>
        <p className="mt-2 text-sm">
          <span
            data-testid="confidence"
            className={cn("inline-flex rounded-full border-2 px-2.5 py-0.5 text-xs font-semibold", BADGE[conf.level])}
          >
            {conf.level} confidence
          </span>{" "}
          <span className="text-muted-foreground">because {conf.reason}</span>
        </p>
        {usesTestBackends(result) && (
          <p className="mt-2 max-w-xl text-sm text-muted-foreground">
            This example uses a simple colour detector and known camera positions. It shows how merging works, not how
            accurate CountOnce is on real photos.
          </p>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" onClick={() => window.print()}>
          Print
        </Button>
        <Button type="button" variant="outline" onClick={downloadCsv}>
          Download CSV
        </Button>
        <Button type="button" variant="outline" onClick={share}>
          {shared ? "Link copied" : "Copy link"}
        </Button>
      </div>
    </section>
  );
}
