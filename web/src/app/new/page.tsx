import { PageShell } from "@/components/chrome/site-chrome";
import { NewJobForm } from "@/components/new-job/new-job-form";
import { requireSession } from "@/lib/server/require-session";

export const metadata = { title: "New count · CountOnce" };
export const dynamic = "force-dynamic";

export default async function NewPage() {
  await requireSession("/new");
  return (
    <PageShell>
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-8">
        <h1 className="text-2xl font-semibold">New count</h1>
        <NewJobForm />
      </div>
    </PageShell>
  );
}
