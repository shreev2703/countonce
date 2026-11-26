"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api/browser";

export function FeedbackForm({ jobId, count }: { jobId: string; count: number }) {
  const [correct, setCorrect] = useState<boolean | null>(null);
  const [actual, setActual] = useState("");
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");

  const send = async () => {
    if (correct === null) return;
    setState("sending");
    try {
      await api.feedback(jobId, {
        correct,
        user_count: !correct && actual !== "" ? Number(actual) : null,
        comment: comment.trim() || null,
      });
      setState("sent");
    } catch {
      setState("error");
    }
  };

  if (state === "sent") {
    return (
      <Card role="status" data-testid="feedback-thanks">
        Thanks, that helps.
      </Card>
    );
  }
  return (
    <Card aria-label="Feedback" data-testid="feedback">
      <h3 className="font-semibold">Is {count} right?</h3>
      <div className="mt-2 flex gap-2" role="group" aria-label="Is the count right?">
        <Button type="button" variant={correct === true ? "default" : "outline"} aria-pressed={correct === true} onClick={() => setCorrect(true)}>
          Yes
        </Button>
        <Button type="button" variant={correct === false ? "default" : "outline"} aria-pressed={correct === false} onClick={() => setCorrect(false)}>
          No
        </Button>
      </div>
      {correct === false && (
        <label className="mt-3 flex flex-col gap-1 text-sm font-medium">
          How many are there, from the photos?
          <input type="number" min={0} value={actual} onChange={(e) => setActual(e.target.value)} className="h-11 w-32 rounded-md border border-border bg-background px-3 font-normal" />
        </label>
      )}
      {correct !== null && (
        <label className="mt-3 flex flex-col gap-1 text-sm font-medium">
          Anything else? (optional)
          <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={2} className="rounded-md border border-border bg-background px-3 py-2 font-normal" />
        </label>
      )}
      <Button type="button" className="mt-3" disabled={correct === null || state === "sending"} onClick={() => void send()}>
        Send feedback
      </Button>
      {state === "error" && <p role="alert" className="mt-2 text-sm">Could not send. Try again.</p>}
    </Card>
  );
}
