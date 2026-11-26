"use client";

import { useEffect, useState } from "react";

import { api, BFF, TERMINAL, type JobView } from "@/lib/api/browser";

export type Transport = "sse" | "polling" | "idle";

/**
 * Live job state: Server-Sent Events first; if the stream errors (proxy that buffers, flaky
 * network) fall back to polling `GET /jobs/{id}` every `pollMs` until the job is finished.
 */
export function useJob(id: string, initial?: JobView, pollMs = 2000): { job: JobView | undefined; transport: Transport; error: string | null } {
  const [job, setJob] = useState<JobView | undefined>(initial);
  const [transport, setTransport] = useState<Transport>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let es: EventSource | undefined;
    const apply = (j: JobView) => {
      setJob(j);
      setError(null);
      return TERMINAL.includes(j.status);
    };
    const poll = async () => {
      if (stopped) return;
      setTransport("polling");
      try {
        if (apply(await api.getJob(id))) return setTransport("idle");
      } catch (e) {
        setError((e as Error).message);
      }
      timer = setTimeout(() => void poll(), pollMs);
    };
    const open = () => {
      if (typeof EventSource === "undefined") return void poll();
      es = new EventSource(`${BFF}/jobs/${id}/events`);
      setTransport("sse");
      const onMsg = (ev: MessageEvent) => apply(JSON.parse(ev.data as string) as JobView);
      es.addEventListener("progress", onMsg);
      es.addEventListener("done", (ev) => {
        onMsg(ev as MessageEvent);
        es?.close();
        setTransport("idle");
      });
      es.onerror = () => {
        es?.close();
        if (!stopped) void poll();
      };
    };
    open();
    return () => {
      stopped = true;
      es?.close();
      if (timer) clearTimeout(timer);
    };
  }, [id, pollMs]);

  return { job, transport, error };
}
