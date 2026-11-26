import { PageShell } from "@/components/chrome/site-chrome";
import { JobView } from "@/components/job/job-view";
import { fetchJobBundle } from "@/lib/server/job";
import { requireSession } from "@/lib/server/require-session";

export const metadata = { title: "Count · CountOnce" };
export const dynamic = "force-dynamic";

export default async function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await requireSession(`/jobs/${id}`);
  const bundle = await fetchJobBundle(id, session.apiKey);
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-8">
        <JobView id={id} initial={bundle.job} initialResult={bundle.result} initialCorrections={bundle.corrections} />
      </div>
    </PageShell>
  );
}
