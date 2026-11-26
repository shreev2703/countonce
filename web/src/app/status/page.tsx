import type { Metadata } from "next";

import { PageShell } from "@/components/chrome/site-chrome";
import { StatusPanel } from "@/components/status/status-panel";

export const metadata: Metadata = {
  title: "Status · CountOnce",
  description: "Whether the processing worker is online, the queue length and recent job times.",
};

export default function StatusPage() {
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-3xl font-semibold">Status</h1>
        <p className="mt-2 text-muted-foreground">
          Photos are processed by a single GPU worker that is not always switched on. This page refreshes by itself.
        </p>
        <StatusPanel />
      </div>
    </PageShell>
  );
}
