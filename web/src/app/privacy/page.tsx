import type { Metadata } from "next";

import { PageShell } from "@/components/chrome/site-chrome";

export const metadata: Metadata = {
  title: "Privacy · CountOnce",
  description: "What CountOnce stores, for how long, and what is removed from your uploads.",
};

/** Describes what the code does today. Update it when retention, blurring or metadata handling change. */
const SECTIONS: { title: string; points: string[] }[] = [
  {
    title: "Browsing",
    points: [
      "The landing page, demo gallery and benchmark need no account.",
      "This site runs no advertising or analytics scripts. The companies hosting it may keep ordinary request logs.",
    ],
  },
  {
    title: "Signing in",
    points: [
      "Sign-in uses Google. The counting service stores an account identifier derived from your Google account.",
      "Your name and email address are kept in an encrypted cookie in your browser for up to 14 days, not in the job database.",
    ],
  },
  {
    title: "Photos and video",
    points: [
      "Photos are re-encoded when the upload completes, which removes embedded metadata such as GPS location and camera details, before they are stored with your job.",
      "Video files are checked but stored as uploaded, so any metadata inside a video, including location, is kept. Remove it first if that matters.",
      "Faces and licence plates are NOT blurred automatically. Do not upload photos of people or plates you would not want stored.",
      "Processing runs on the project owner’s GPU computer, which can therefore access uploaded files.",
    ],
  },
  {
    title: "How long it is kept",
    points: [
      "Uploads and results of finished jobs are deleted automatically after 30 days. The cleanup runs inside the processing worker, so it happens while that worker is running; if it is switched off, files stay until you delete them or the owner clears old jobs.",
      "You can delete a job from History once it is not running. This removes its uploaded files and results and stops its share links working.",
      "A record that the job existed (including what you asked to count and when), and any feedback or corrections you sent, is kept after deletion.",
    ],
  },
  {
    title: "Sharing",
    points: ["Anyone who has a share link can see that result and what was counted, without signing in."],
  },
  {
    title: "Training",
    points: [
      "Your uploads are not used to train models without your permission.",
      "Corrections are marked as usable for improving CountOnce only if you tick the box when you send them.",
    ],
  },
];

export default function Privacy() {
  return (
    <PageShell>
      <div className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-3xl font-semibold">Privacy</h1>
        <p className="mt-2 text-muted-foreground">What happens to your data today, in plain words.</p>
        {SECTIONS.map((s) => (
          <section key={s.title} className="mt-8">
            <h2 className="text-xl font-semibold">{s.title}</h2>
            <ul className="mt-2 list-disc pl-5">
              {s.points.map((p) => (
                <li key={p} className="mt-1">{p}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </PageShell>
  );
}
