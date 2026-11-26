import { PageShell } from "@/components/chrome/site-chrome";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { safeNext } from "@/lib/safe-next";
import { devAuthEnabled, googleConfig } from "@/lib/server/config";

export const metadata = { title: "Sign in · CountOnce" };
export const dynamic = "force-dynamic";

const ERRORS: Record<string, string> = {
  google_not_configured: "Google sign-in is not set up on this server.",
  oauth_state: "The sign-in took too long or was interrupted. Try again.",
  oauth_exchange: "Google did not accept the sign-in. Try again.",
  oauth_claims: "Google returned an unexpected account. Try again.",
};

export default async function SignInPage({ searchParams }: { searchParams: Promise<{ next?: string; error?: string }> }) {
  const { next, error } = await searchParams;
  const dest = safeNext(next);
  const google = googleConfig() !== null;
  const dev = devAuthEnabled();
  return (
    <PageShell>
      <div className="mx-auto flex max-w-md flex-col gap-4 px-4 py-12">
        <h1 className="text-2xl font-semibold">Sign in</h1>
        <p className="text-muted-foreground">Sign in to count your own photos and keep a history. The demo needs no account.</p>
        {error && (
          <p role="alert" className="rounded-md border border-status-rejected p-3 text-sm">
            {ERRORS[error] ?? "Sign-in failed."}
          </p>
        )}
        <Card className="flex flex-col gap-3">
          {google ? (
            <Button asChild size="lg">
              <a href={`/api/auth/google?next=${encodeURIComponent(dest)}`}>Continue with Google</a>
            </Button>
          ) : (
            <Button size="lg" disabled>
              Continue with Google (not configured)
            </Button>
          )}
          {dev && (
            <form action="/api/auth/dev" method="post" className="flex flex-col gap-2 border-t border-border pt-3" data-testid="dev-signin">
              <p className="text-sm text-muted-foreground">Test sign-in (development only)</p>
              <input type="hidden" name="next" value={dest} />
              <input name="user" defaultValue="tester" aria-label="Test user name" className="h-11 rounded-md border border-border bg-background px-3" />
              <Button type="submit" variant="outline">
                Sign in as test user
              </Button>
            </form>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
