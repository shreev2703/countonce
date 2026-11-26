"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api/browser";
import { objectLabel } from "@/lib/result-view/model";
import type { Result } from "@/lib/schemas/result";

/**
 * Corrections on the stored result: "not an object" (reject), "same object" (merge two
 * instances) and "different objects" (split some views off one). The server replays them on the
 * original result, so nothing is lost and repeating one does nothing.
 */
export function CorrectionsPanel({ jobId, result, onChanged }: { jobId: string; result: Result; onChanged: () => void }) {
  const [picked, setPicked] = useState<number[]>([]);
  const [splitting, setSplitting] = useState<number | null>(null);
  const [members, setMembers] = useState<string[]>([]);
  const [consent, setConsent] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async (action: "reject" | "merge" | "split", detection_ids: string[]) => {
    setBusy(true);
    try {
      const r = await api.correct(jobId, { action, detection_ids, consent });
      setMsg({ ok: true, text: r.idempotent_replay ? "Already applied." : `Done. The count is now ${r.corrected_count}.` });
      setPicked([]);
      setSplitting(null);
      setMembers([]);
      onChanged();
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const byId = new Map(result.instances.map((i) => [i.id, i]));
  const mergeIds = picked.map((id) => byId.get(id)?.members[0]).filter((m): m is string => Boolean(m));

  return (
    <Card aria-label="Corrections" data-testid="corrections">
      <h3 className="font-semibold">Something wrong? Correct it</h3>
      <p className="text-sm text-muted-foreground">Pick what the count got wrong. Your original result is kept.</p>
      <ul className="mt-3 flex flex-col gap-2">
        {result.instances.map((i) => (
          <li key={i.id} data-testid={`instance-${i.id}`} className="flex flex-wrap items-center gap-2 rounded-md border border-border p-2 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={picked.includes(i.id)} onChange={(e) => setPicked((p) => (e.target.checked ? [...p, i.id] : p.filter((x) => x !== i.id)))} />
              <span className="font-medium">{objectLabel(i.id)}</span>
              <span className="text-muted-foreground">seen in {i.n_views} photo{i.n_views === 1 ? "" : "s"}</span>
            </label>
            <span className="ml-auto flex gap-2">
              <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => void run("reject", i.members)}>
                Not an object
              </Button>
              {i.members.length > 1 && (
                <Button type="button" size="sm" variant="outline" onClick={() => { setSplitting(splitting === i.id ? null : i.id); setMembers([]); }}>
                  Split…
                </Button>
              )}
            </span>
            {splitting === i.id && (
              <fieldset className="basis-full">
                <legend className="text-xs text-muted-foreground">Tick the photos that show a different object</legend>
                {i.members.map((m) => (
                  <label key={m} className="mr-3 inline-flex items-center gap-1">
                    <input type="checkbox" checked={members.includes(m)} onChange={(e) => setMembers((cur) => (e.target.checked ? [...cur, m] : cur.filter((x) => x !== m)))} />
                    photo {Number(m.split(":")[0]) + 1}
                  </label>
                ))}
                <Button type="button" size="sm" disabled={busy || members.length === 0 || members.length >= i.members.length} onClick={() => void run("split", members)}>
                  Split off
                </Button>
              </fieldset>
            )}
          </li>
        ))}
      </ul>
      <div className="mt-3 flex flex-col gap-2">
        <Button type="button" variant="outline" className="self-start" disabled={busy || picked.length < 2} onClick={() => void run("merge", mergeIds)}>
          Merge {picked.length || ""} selected (same object)
        </Button>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          Let CountOnce use my corrections to improve (optional).
        </label>
      </div>
      {msg && (
        <p role={msg.ok ? "status" : "alert"} data-testid="correction-msg" className="mt-2 text-sm">
          {msg.text}
        </p>
      )}
    </Card>
  );
}
