"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";

/**
 * 3D viewer frame (react-three-fiber + drei). Scaffold only: the point cloud, markers and frusta
 * arrive with M14. Load it lazily (next/dynamic) so it stays out of the initial JS budget.
 */
export default function ViewerFrame({ label }: { label: string }) {
  return (
    <div role="img" aria-label={label} className="h-72 w-full overflow-hidden rounded-lg bg-muted">
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
        <ambientLight intensity={0.8} />
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#4f6bd8" />
        </mesh>
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
