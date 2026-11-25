import { cn } from "@/lib/utils";

/** Detection status. Colour is redundant with the label and the outline style (PLAN 17.10). */
export type DetectionStatus = "new" | "duplicate" | "rejected";

const STYLES: Record<DetectionStatus, { label: string; className: string }> = {
  new: { label: "Counted", className: "border-2 border-solid text-status-new" },
  duplicate: { label: "Duplicate", className: "border-2 border-dashed text-status-duplicate" },
  rejected: { label: "Rejected", className: "border-2 border-dotted text-status-rejected" },
};

export function StatusBadge({ status, className }: { status: DetectionStatus; className?: string }) {
  const s = STYLES[status];
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        s.className,
        className,
      )}
    >
      {s.label}
    </span>
  );
}
