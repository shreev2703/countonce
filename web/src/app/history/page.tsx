import { PageShell } from "@/components/chrome/site-chrome";
import { HistoryList } from "@/components/job/history-list";
import { requireSession } from "@/lib/server/require-session";

export const metadata = { title: "History · CountOnce" };
export const dynamic = "force-dynamic";

export default async function HistoryPage() {
  await requireSession("/history");
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-8">
        <h1 className="mb-4 text-2xl font-semibold">History</h1>
        <HistoryList />
      </div>
    </PageShell>
  );
}
