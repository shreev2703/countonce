import type { Metadata } from "next";

import { PageShell } from "@/components/chrome/site-chrome";

export const metadata: Metadata = {
  title: "About · CountOnce",
  description: "Who builds CountOnce, where the code is and how to get in touch.",
};

const REPO = "https://github.com/shreev2703/countonce";
const LINKS = [
  ["Source code", REPO, "The pipeline, benchmark and this site."],
  ["Component status", `${REPO}/blob/main/docs/STATUS.md`, "What has actually been tested, component by component."],
  ["Sparse-view reconstruction benchmark", "https://github.com/shreev2703/vggtProject", "The earlier VGGT against COLMAP project this builds on."],
  ["Questions and problems", `${REPO}/issues`, "Open an issue to get in touch."],
] as const;

export default function About() {
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-3xl font-semibold">About</h1>
        <p className="mt-3">
          CountOnce is a research project by Shreevatson Babu Sreegeetha. It counts objects from a few photos by
          placing every detection in one 3D scene, so that an object seen in several photos is counted once.
        </p>
        <p className="mt-3 text-muted-foreground">
          It is not a finished product. Counts can be wrong, sometimes badly; see the benchmark before relying on one.
          There is no paper yet.
        </p>
        <ul className="mt-6 flex flex-col gap-4">
          {LINKS.map(([label, href, body]) => (
            <li key={href}>
              <a href={href} className="font-medium underline underline-offset-4">{label}</a>
              <p className="text-sm text-muted-foreground">{body}</p>
            </li>
          ))}
        </ul>
      </div>
    </PageShell>
  );
}
