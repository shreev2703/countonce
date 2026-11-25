"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { DemoScene } from "@/lib/result-view/demo";
import { confidenceBadge, csvForResult, intervalText } from "@/lib/result-view/model";
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
    <Card aria-labelledby="count-heading" className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h2 id="count-heading" className="sr-only">
          Count result
        </h2>
        <p className="text-sm text-muted-foreground">{result.prompt}</p>
        <p
          data-testid="count"
          className="text-6xl font-semibold leading-none tabular-nums"
          aria-label={`Count: ${result.count} ${result.prompt}, ${intervalText(result)}`}
        >
          {result.count}
        </p>
        <p className="mt-2 text-sm" data-testid="interval">
          {intervalText(result)}
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
      </div>
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" onClick={() => window.print()}>
          Report
        </Button>
        <Button type="button" variant="outline" onClick={downloadCsv}>
          CSV
        </Button>
        <Button type="button" variant="outline" onClick={share}>
          {shared ? "Link copied" : "Share"}
        </Button>
      </div>
    </Card>
  );
}
