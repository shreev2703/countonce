"use client";

import { Component, type ReactNode } from "react";

/** Without WebGL the 3D renderer throws; keep the count, photos and tabs usable. */
export class ViewerBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  override render() {
    if (!this.state.failed) return this.props.children;
    return (
      <p role="status" className="grid h-full place-items-center p-6 text-center text-sm text-muted-foreground">
        The 3D view could not start on this device (it needs WebGL). The count, photos and explanations still work.
      </p>
    );
  }
}
