"use client";

import { useState } from "react";

import type { FujiCell } from "@/lib/benchmark";

/** Fixed order: a method keeps its colour and shape when others are filtered out. */
const METHODS = [
  { id: "countonce_v1", label: "CountOnce v1", color: "var(--series-1)" },
  { id: "countonce_v2", label: "CountOnce v2", color: "var(--series-2)" },
  { id: "sum", label: "Sum of photos", color: "var(--series-3)" },
  { id: "max", label: "Max of photos", color: "var(--series-4)" },
  { id: "median", label: "Median of photos", color: "var(--series-5)" },
];

const METRICS = [
  { id: "mae", label: "Mean absolute error (apples)", digits: 1, note: "Lower is better." },
  { id: "signed_bias", label: "Signed bias (apples)", digits: 1, note: "Predicted minus true: closer to zero is better." },
  { id: "mape", label: "Mean absolute percentage error (%)", digits: 1, note: "Lower is better. Over 100% means the error is larger than the true number." },
];

const W = 720;
const H = 320;
const M = { left: 52, right: 12, top: 12, bottom: 40 };

function ticks(lo: number, hi: number): number[] {
  const raw = (hi - lo) / 5 || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.floor(lo / step) * step; v < hi + step - 1e-9; v += step) out.push(v);
  return out;
}

export function BenchmarkExplorer({ cells }: { cells: FujiCell[] }) {
  const [metric, setMetric] = useState("mae");
  const [hidden, setHidden] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);

  const info = METRICS.find((m) => m.id === metric) ?? METRICS[0]!;
  const shown = METHODS.filter((m) => !hidden.includes(m.id));
  const data = cells.filter((c) => c.metric === metric);
  const ks = [...new Set(data.map((c) => c.k))].sort((a, b) => a - b);
  const find = (method: string, k: number) => data.find((c) => c.method === method && c.k === k);
  const visible = data.filter((c) => !hidden.includes(c.method));
  const yt = ticks(
    Math.min(0, ...visible.map((c) => c.low ?? c.estimate)),
    Math.max(1, ...visible.map((c) => c.high ?? c.estimate)),
  );
  const y0 = yt[0]!;
  const y1 = yt[yt.length - 1]!;
  const y = (v: number) => M.top + ((y1 - v) / (y1 - y0)) * (H - M.top - M.bottom);
  const band = (W - M.left - M.right) / Math.max(1, ks.length);
  const x = (ki: number, mi: number) => M.left + band * ki + (band * (mi + 1)) / (METHODS.length + 1);
  const f = (v: number) => v.toFixed(info.digits);
  const text = (c: FujiCell) => (c.low === null || c.high === null ? f(c.estimate) : `${f(c.estimate)} [${f(c.low)}, ${f(c.high)}]`);

  return (
    <div className="viz">
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1 text-sm font-medium">
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)} className="h-10 rounded-md border border-border bg-card px-3 font-normal">
            {METRICS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <fieldset className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
          <legend className="mb-1 font-medium">Methods</legend>
          {METHODS.map((m) => (
            <label key={m.id} className="flex min-h-6 items-center gap-2">
              <input
                type="checkbox"
                checked={!hidden.includes(m.id)}
                onChange={(e) => setHidden((h) => (e.target.checked ? h.filter((x) => x !== m.id) : [...h, m.id]))}
              />
              <span aria-hidden className="inline-block size-3 rounded-full" style={{ background: m.color }} />
              {m.label}
            </label>
          ))}
        </fieldset>
      </div>

      <p className="mt-3 text-sm text-muted-foreground">
        {info.note} Dots are estimates over test trees; whiskers are 95% bootstrap intervals.
      </p>
      <div className="mt-2 overflow-x-auto" tabIndex={0} role="region" aria-label="Benchmark chart">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[520px]" role="img" aria-label={`${info.label} by number of photos and method. The table below has the same values.`}>
          {yt.map((v) => (
            <g key={v}>
              <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeWidth={v === 0 ? 2 : 1} />
              <text x={M.left - 8} y={y(v) + 4} textAnchor="end" fontSize={12} fill="var(--muted-foreground)">
                {v}
              </text>
            </g>
          ))}
          {ks.map((k, ki) => (
            <g key={k}>
              <text x={M.left + band * (ki + 0.5)} y={H - 18} textAnchor="middle" fontSize={12} fill="var(--muted-foreground)">
                {k} photos
              </text>
              {METHODS.map((m, mi) => {
                const c = find(m.id, k);
                if (!c || hidden.includes(m.id)) return null;
                const cx = x(ki, mi);
                const key = `${m.id}|${k}`;
                const label = `${m.label}, ${k} photos: ${text(c)}`;
                return (
                  <g key={m.id} onPointerEnter={() => setActive(label)} onPointerLeave={() => setActive(null)} data-testid={`dot-${key}`}>
                    <title>{label}</title>
                    <rect x={cx - band / 12} y={M.top} width={band / 6} height={H - M.top - M.bottom} fill="transparent" />
                    {c.low !== null && c.high !== null && <line x1={cx} x2={cx} y1={y(c.low)} y2={y(c.high)} stroke={m.color} strokeWidth={2} />}
                    <circle cx={cx} cy={y(c.estimate)} r={5} fill={m.color} stroke="var(--background)" strokeWidth={2} />
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
      <p aria-live="polite" className="tabular min-h-6 text-sm">
        {active ?? "Hover a dot for its exact value."}
      </p>

      <div className="mt-3 overflow-x-auto" tabIndex={0} role="region" aria-label="Benchmark values">
        <table className="tabular w-full text-sm">
          <caption className="sr-only">{info.label}: estimate [95% interval]</caption>
          <thead>
            <tr className="border-b border-border text-left">
              <th scope="col" className="py-2 pr-4">Method</th>
              {ks.map((k) => (
                <th key={k} scope="col" className="py-2 pr-4">
                  {k} photos
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((m) => (
              <tr key={m.id} className="border-b border-border">
                <th scope="row" className="py-2 pr-4 text-left font-medium">{m.label}</th>
                {ks.map((k) => {
                  const c = find(m.id, k);
                  return (
                    <td key={k} className="whitespace-nowrap py-2 pr-4">
                      {c ? text(c) : "not measured"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
