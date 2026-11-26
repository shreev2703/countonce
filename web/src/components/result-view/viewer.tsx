"use client";

import { Billboard, Line, OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import {
  cameraFovDeg,
  frustumCorners,
  lodStride,
  objectLabel,
  sceneBounds,
  type CameraPose,
  type Detection,
  type Vec3,
} from "@/lib/result-view/model";
import type { MarkersDoc } from "@/lib/schemas/result";

export type ViewCommand = { kind: "reset" | "top" | "camera" | "object"; id?: number; n: number };

export interface ViewerProps {
  xyz: Float32Array;
  rgb: Uint8Array;
  maxPoints: number;
  markers: MarkersDoc["markers"];
  cameras: CameraPose[];
  rejected: Detection[];
  showRejected: boolean;
  selectedObject: number | null;
  highlightObject: number | null;
  selectedCamera: number | null;
  command: ViewCommand;
  reducedMotion: boolean;
  onSelectObject: (id: number | null) => void;
  onHoverObject: (id: number | null) => void;
  onSelectCamera: (view: number) => void;
  onReady: (gl: THREE.WebGLRenderer) => void;
}

const UP: Vec3 = [0, 0, 1];

function PointCloud({ xyz, rgb, maxPoints }: Pick<ViewerProps, "xyz" | "rgb" | "maxPoints">) {
  const geometry = useMemo(() => {
    const n = xyz.length / 3;
    const stride = lodStride(n, maxPoints);
    const m = Math.ceil(n / stride);
    const pos = new Float32Array(m * 3);
    const col = new Float32Array(m * 3);
    for (let i = 0, j = 0; i < n; i += stride, j++) {
      for (let k = 0; k < 3; k++) {
        pos[j * 3 + k] = xyz[i * 3 + k] ?? 0;
        col[j * 3 + k] = (rgb[i * 3 + k] ?? 0) / 255;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [xyz, rgb, maxPoints]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.035} sizeAttenuation vertexColors />
    </points>
  );
}

function Marker({
  m,
  active,
  selected,
  dimmed,
  reducedMotion,
  onSelect,
  onHover,
}: {
  m: MarkersDoc["markers"][number];
  active: boolean;
  selected: boolean;
  dimmed: boolean;
  reducedMotion: boolean;
  onSelect: () => void;
  onHover: (on: boolean) => void;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const s = active && !reducedMotion ? 1 + 0.15 * Math.sin(clock.elapsedTime * 6) : 1;
    ref.current.scale.setScalar(s);
  });
  const color = selected ? "#f59e0b" : active ? "#fbbf24" : "#16a34a";
  return (
    <group position={m.xyz}>
      <mesh
        ref={ref}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(true);
        }}
        onPointerOut={() => onHover(false)}
      >
        <sphereGeometry args={[Math.max(m.size / 2, 0.05), 24, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.25} transparent opacity={dimmed ? 0.3 : 0.85} />
      </mesh>
      <Billboard follow lockZ={false}>
        <mesh position={[0, 0, 0.001]} renderOrder={10}>
          <planeGeometry args={[Math.max(m.size, 0.12) * 0.9, Math.max(m.size, 0.12) * 0.45]} />
          <meshBasicMaterial color="#000000" transparent opacity={0.75} depthTest={false} />
        </mesh>
        <Text
          fontSize={Math.max(m.size, 0.12) * 0.3}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          renderOrder={11}
          material-depthTest={false}
        >
          {objectLabel(m.id)}
        </Text>
      </Billboard>
    </group>
  );
}

function Frustum({
  cam,
  depth,
  selected,
  onSelect,
}: {
  cam: CameraPose;
  depth: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const corners = useMemo(() => frustumCorners(cam, depth), [cam, depth]);
  const color = selected ? "#f59e0b" : "#6b7280";
  const edges: Vec3[][] = [...corners.map((c) => [cam.center, c]), [...corners, ...corners.slice(0, 1)]];
  return (
    <group>
      {edges.map((pts, i) => (
        <Line key={i} points={pts} color={color} lineWidth={selected ? 2.5 : 1.2} />
      ))}
      <mesh
        position={cam.center}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      >
        <sphereGeometry args={[depth * 0.18, 12, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}

const FLIGHT_SECONDS = 0.9;

interface Flight {
  fromPos: THREE.Vector3;
  fromLook: THREE.Vector3;
  fromFov: number;
  pos: THREE.Vector3;
  look: THREE.Vector3;
  fov: number;
  t: number;
}

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Flies the camera and orbit target to the last command (object, camera, top view, reset). */
/* r3f camera/controls are mutable by design; the react-hooks immutability rule does not apply. */
/* eslint-disable react-hooks/immutability */
function CameraRig({
  command,
  cameras,
  markers,
  center,
  radius,
  reducedMotion,
}: {
  command: ViewCommand;
  cameras: CameraPose[];
  markers: MarkersDoc["markers"];
  center: Vec3;
  radius: number;
  reducedMotion: boolean;
}) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null;
  const flight = useRef<Flight | null>(null);

  // Dragging or zooming takes over from a flight in progress.
  useEffect(() => {
    if (!controls) return;
    const cancel = () => {
      flight.current = null;
    };
    controls.addEventListener("start", cancel);
    return () => controls.removeEventListener("start", cancel);
  }, [controls]);

  useEffect(() => {
    const c = new THREE.Vector3(...center);
    const from = {
      fromPos: camera.position.clone(),
      fromLook: controls?.target.clone() ?? c.clone(),
      fromFov: camera.fov,
      t: 0,
    };
    let goal: { pos: THREE.Vector3; look: THREE.Vector3; fov: number } | null = null;
    if (command.kind === "reset") {
      goal = { pos: c.clone().add(new THREE.Vector3(1.4, -1.6, 1.1).multiplyScalar(radius)), look: c, fov: 50 };
    } else if (command.kind === "top") {
      goal = { pos: c.clone().add(new THREE.Vector3(0, -0.02, 2.8 * radius)), look: c, fov: 50 };
    } else if (command.kind === "camera") {
      const cam = cameras.find((x) => x.view === command.id);
      if (cam) {
        const pos = new THREE.Vector3(...cam.center);
        const look = pos.clone().add(new THREE.Vector3(...cam.forward).multiplyScalar(pos.distanceTo(c)));
        goal = { pos, look, fov: cameraFovDeg(cam) };
      }
    } else if (command.kind === "object") {
      const m = markers.find((x) => x.id === command.id);
      if (m) {
        // Keep the viewing direction, centre on the object and move in closer.
        const look = new THREE.Vector3(...m.xyz);
        const offset = from.fromPos.clone().sub(from.fromLook);
        offset.setLength(THREE.MathUtils.clamp(offset.length(), radius * 0.7, radius * 1.3));
        goal = { pos: look.clone().add(offset), look, fov: 50 };
      }
    }
    flight.current = goal && { ...from, ...goal };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [command]);

  useFrame((_, dt) => {
    const f = flight.current;
    if (!f || !controls) return;
    // dt is capped so a stalled first frame does not skip the whole flight.
    f.t = reducedMotion ? 1 : Math.min(1, f.t + Math.min(dt, 0.05) / FLIGHT_SECONDS);
    const e = easeInOut(f.t);
    camera.position.lerpVectors(f.fromPos, f.pos, e);
    controls.target.lerpVectors(f.fromLook, f.look, e);
    camera.fov = THREE.MathUtils.lerp(f.fromFov, f.fov, e);
    camera.updateProjectionMatrix();
    controls.update();
    if (f.t >= 1) flight.current = null;
  });
  return null;
}

/* eslint-enable react-hooks/immutability */

function Ready({ onReady }: { onReady: (gl: THREE.WebGLRenderer) => void }) {
  const gl = useThree((s) => s.gl);
  useEffect(() => onReady(gl), [gl, onReady]);
  return null;
}

export default function Viewer(props: ViewerProps) {
  const { xyz, markers, cameras, rejected, showRejected, selectedObject, highlightObject, selectedCamera } = props;
  const { center, radius } = useMemo(() => sceneBounds(xyz, markers), [xyz, markers]);
  const frustumDepth = radius * 0.22;
  const lineObject = highlightObject ?? selectedObject;
  const lineMarker = markers.find((m) => m.id === lineObject);
  const lineViews = useMemo(
    () => (lineMarker ? [...new Set(lineMarker.members.map((id) => Number(id.split(":")[0])))] : []),
    [lineMarker],
  );
  const start = useMemo(
    () => new THREE.Vector3(...center).add(new THREE.Vector3(1.4, -1.6, 1.1).multiplyScalar(radius)).toArray(),
    [center, radius],
  );

  return (
    <Canvas
      camera={{ position: start as Vec3, fov: 50, up: UP, near: 0.01, far: radius * 40 }}
      gl={{ preserveDrawingBuffer: true }}
      onPointerMissed={() => props.onSelectObject(null)}
    >
      <color attach="background" args={["#0b0d13"]} />
      <Ready onReady={props.onReady} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, -4, 6]} intensity={1.2} />
      <PointCloud xyz={props.xyz} rgb={props.rgb} maxPoints={props.maxPoints} />
      {markers.map((m) => (
        <Marker
          key={m.id}
          m={m}
          active={m.id === highlightObject}
          selected={m.id === selectedObject}
          dimmed={(highlightObject ?? selectedObject) !== null && m.id !== highlightObject && m.id !== selectedObject}
          reducedMotion={props.reducedMotion}
          onSelect={() => props.onSelectObject(m.id)}
          onHover={(on) => props.onHoverObject(on ? m.id : null)}
        />
      ))}
      {showRejected &&
        rejected
          .filter((d) => d.xyz)
          .map((d) => (
            <mesh key={d.id} position={d.xyz as Vec3}>
              <sphereGeometry args={[Math.max((d.size ?? 0.2) / 2, 0.04), 12, 8]} />
              <meshStandardMaterial color="#dc2626" transparent opacity={0.25} />
            </mesh>
          ))}
      {cameras.map((c) => (
        <Frustum
          key={c.view}
          cam={c}
          depth={frustumDepth}
          selected={c.view === selectedCamera}
          onSelect={() => props.onSelectCamera(c.view)}
        />
      ))}
      {lineMarker &&
        lineViews.map((v) => {
          const cam = cameras.find((c) => c.view === v);
          return cam ? (
            <Line key={v} points={[lineMarker.xyz as Vec3, cam.center]} color="#f59e0b" lineWidth={1.5} dashed dashSize={0.1} gapSize={0.06} />
          ) : null;
        })}
      <OrbitControls makeDefault enablePan={false} />
      <CameraRig
        command={props.command}
        cameras={cameras}
        markers={markers}
        center={center}
        radius={radius}
        reducedMotion={props.reducedMotion}
      />
    </Canvas>
  );
}
