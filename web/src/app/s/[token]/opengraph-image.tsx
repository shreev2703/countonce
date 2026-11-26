import { ImageResponse } from "next/og";

import { fetchShare } from "@/lib/server/share";

export const alt = "CountOnce result";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const dynamic = "force-dynamic";

export default async function Image({ params }: { params: Promise<{ token: string }> }) {
  const v = await fetchShare((await params).token);
  const r = v?.result;
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between", background: "#0f172a", color: "white", padding: 72, fontFamily: "sans-serif" }}>
        <div style={{ display: "flex", fontSize: 36, opacity: 0.8 }}>CountOnce</div>
        {r ? (
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", fontSize: 200, fontWeight: 700, lineHeight: 1 }}>{r.count}</div>
            <div style={{ display: "flex", fontSize: 48, marginTop: 16 }}>{v.prompt}</div>
            <div style={{ display: "flex", fontSize: 32, marginTop: 12, opacity: 0.8 }}>
              {`likely ${r.interval.low}–${r.interval.high}, counted from photos`}
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", fontSize: 64 }}>Shared count not available</div>
        )}
        <div style={{ display: "flex", fontSize: 28, opacity: 0.6 }}>Each object counted once</div>
      </div>
    ),
    size,
  );
}
