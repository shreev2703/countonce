import Link from "next/link";

import { SpotlightCard } from "@/components/motion/spotlight-card";
import type { DemoCard } from "@/lib/demo-index";

export function DemoSceneCard({ scene }: { scene: DemoCard }) {
  return (
    <SpotlightCard className="[&>div:last-child]:gap-3">
      {scene.thumb ? (
        // eslint-disable-next-line @next/next/no-img-element -- tiny static thumbnail
        <img
          src={scene.thumb}
          alt={`Photo 1 of ${scene.nFrames} from the ${scene.label}, with detections drawn on it`}
          width={160}
          height={120}
          loading="lazy"
          decoding="async"
          className="aspect-[4/3] w-full rounded-md border border-border object-cover [image-rendering:pixelated]"
        />
      ) : null}
      <div>
        <h3 className="font-semibold">
          <Link href={`/demo/${scene.id}`} className="underline-offset-4 hover:underline">
            {scene.label}
          </Link>
        </h3>
        <p className="text-sm text-muted-foreground">
          Counting &ldquo;{scene.prompt}&rdquo; from {scene.nFrames} photos
        </p>
      </div>
      <p className="tabular text-sm">
        <span className="text-2xl font-semibold">{scene.count}</span> counted
        {scene.gtCount !== null ? <span className="text-muted-foreground"> · {scene.synthetic ? "known true count" : "hand count"} {scene.gtCount}</span> : null}
      </p>
      {scene.synthetic ? (
        <p className="text-xs text-muted-foreground">
          Synthetic test scene, not a real-world capture.
        </p>
      ) : null}
    </SpotlightCard>
  );
}
