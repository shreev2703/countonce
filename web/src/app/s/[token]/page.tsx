import type { Metadata } from "next";

import { PageShell } from "@/components/chrome/site-chrome";
import { Card } from "@/components/ui/card";
import { confidenceBadge, intervalText } from "@/lib/result-view/model";
import { fetchShare } from "@/lib/server/share";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: { params: Promise<{ token: string }> }): Promise<Metadata> {
  const v = await fetchShare((await params).token);
  if (!v?.result) return { title: "Shared count · CountOnce", robots: { index: false } };
  const title = `${v.result.count} ${v.prompt} · CountOnce`;
  const description = `Counted from photos, each object once: ${intervalText(v.result)}.`;
  return { title, description, robots: { index: false }, openGraph: { title, description, type: "website" }, twitter: { card: "summary_large_image", title, description } };
}

export default async function SharedPage({ params }: { params: Promise<{ token: string }> }) {
  const v = await fetchShare((await params).token);
  if (!v || !v.result) {
    return (
      <PageShell>
        <div className="mx-auto max-w-2xl px-4 py-12" data-testid="share-missing">
          <h1 className="text-2xl font-semibold">This link is not available</h1>
          <p className="mt-2 text-muted-foreground">It may have been turned off by its owner, or the result is gone.</p>
        </div>
      </PageShell>
    );
  }
  const conf = confidenceBadge(v.result);
  return (
    <PageShell>
      <div className="mx-auto flex max-w-2xl flex-col gap-4 px-4 py-8">
        <h1 className="text-2xl font-semibold">Shared count</h1>
        <Card aria-label="Count result">
          <p className="text-sm text-muted-foreground">{v.prompt}</p>
          <p data-testid="shared-count" className="text-6xl font-semibold leading-none tabular-nums">
            {v.result.count}
          </p>
          <p className="mt-2 text-sm" data-testid="shared-interval">{intervalText(v.result)}</p>
          <p className="mt-2 text-sm">
            <span className="font-semibold">{conf.level} confidence</span>{" "}
            <span className="text-muted-foreground">because {conf.reason}</span>
          </p>
        </Card>
        {v.result.warnings.length > 0 && (
          <Card aria-label="Warnings">
            <ul className="list-disc pl-5 text-sm">
              {v.result.warnings.map((w) => (
                <li key={w.code}>{w.message}</li>
              ))}
            </ul>
          </Card>
        )}
        <p className="text-sm text-muted-foreground">
          Only the count and its range are shared, not the photos. The owner can turn this link off at any time.
        </p>
      </div>
    </PageShell>
  );
}
