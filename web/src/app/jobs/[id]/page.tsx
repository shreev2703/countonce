import { PageShell } from "@/components/chrome/site-chrome";
import { JobView } from "@/components/job/job-view";
import { requireSession } from "@/lib/server/require-session";

export const metadata = { title: "Count · CountOnce" };
export const dynamic = "force-dynamic";

export default async function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  await requireSession(`/jobs/${id}`);
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-8">
        <JobView id={id} />
      </div>
    </PageShell>
  );
}
