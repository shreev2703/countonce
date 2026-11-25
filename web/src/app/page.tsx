import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";

// Placeholder shell for the M12 scaffold; the real landing page is M15.
export default function Home() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-12">
      <h1 className="text-3xl font-semibold">CountOnce</h1>
      <p className="mt-2 text-muted-foreground">
        Count objects from a few photos, each one exactly once.
      </p>
      <Card className="mt-8">
        <p className="text-sm text-muted-foreground">Status legend</p>
        <div className="mt-3 flex gap-3">
          <StatusBadge status="new" />
          <StatusBadge status="duplicate" />
          <StatusBadge status="rejected" />
        </div>
      </Card>
    </main>
  );
}
