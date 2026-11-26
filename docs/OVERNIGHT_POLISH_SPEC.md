# CountOnce Overnight Polish & UI Overhaul Spec

## 1. Fix React 19 / Next.js 16 3D Viewer Crash (`U1`)
- In `web/src/components/result-view/viewer.tsx` (around line 109 inside `Marker`), replace `@react-three/drei`'s `<Html>` with WebGL-native `<Billboard>` and `<Text>` (with a crisp dark pill/badge background mesh or outline).
- Must permanently eliminate both Next.js dev errors:
  - `Attempted to synchronously unmount a root while React was already rendering`
  - `Runtime NotFoundError: Failed to execute 'removeChild' on 'Node'`
- Verify zero console errors and zero Next.js error badges on `/demo/tiny`.

## 2. Hide Google Sign-In From UI — Keep All Code (`U1`)
- Do NOT delete any auth routes, OAuth handlers, or sign-in code—keep all of it intact for when OAuth is configured later.
- Remove the "Sign in" tab/link from the top navigation bar and any redundant sign-in CTAs.
- On the landing page (`/`), remove "Counting your own photos needs a Google sign-in" helper text and ensure primary buttons never dead-end on an unconfigured OAuth screen (route cleanly to the interactive demo or a sleek coming-soon state for cloud uploads).

## 3. Flagship UI/UX Redesign & Animations (`U2`)
- **Visual Direction:** Base the aesthetic on **Linear** (`linear.app`) + **Luma AI** (`lumalabs.ai`) + **Raycast** (`raycast.com`):
  - Deep obsidian/zinc layered surfaces (not flat black), 1px translucent hairline borders, crisp Inter/Geist typography, subtle radial spotlight glows, and glassmorphic floating controls over the 3D canvas.
- **Animations:** Install and use `framer-motion` (or `motion`) for smooth, GPU-accelerated animations:
  - Staggered hero reveals, hover spotlight cards, smooth tab indicator glides, and an animated visual diagram on `/` showing multiple overlapping camera frames merging into a single deduplicated 3D count.
- **Landing Page (`/`):**
  - Replace the blurry, pixelated synthetic red-box hero image with a sharp, high-craft visual presentation.
  - Keep all claims strictly honest per `docs/STATUS.md` (do not invent benchmark numbers or pretend synthetic scenes are real photos), just present them cleanly.
- **3D Results Viewer (`/demo/tiny`):**
  - Make it feel like a flagship spatial app: floating translucent toolbar, smooth camera fly-to when clicking an object, sleek collapsible side inspector, and clean two-way highlight between 3D markers and photo thumbnails.
  - Translate raw developer jargon into plain, intuitive language; move technical metadata (`schema 1.1`, commit hashes, raw sensitivity phrasing) into a collapsible Details drawer.

## 4. Prospective-User Playwright Audit (`U3`)
- Start the Next.js server and use Playwright to browse `/`, `/demo/tiny`, `/new`, and `/benchmark` as a first-time non-technical user on both desktop (`1440x900`) and mobile (`390x844`).
- Check browser console logs on every route (must have 0 console errors and 0 hydration/runtime errors).
- Fix any layout, overflow, contrast, or UX friction found during the audit.
- Run `npm run build` and existing frontend tests in `web/` until green.

## 5. Fuji-SfM Collinear Camera Rotation Alignment Fix (`F1`)
- Fix the open alignment bug logged in `docs/GATES.md` ("Fuji window alignment leaves rotation unconstrained"):
  - Update window alignment (`bench/fuji_core.py` / `countonce/recon/align.py`) to incorporate camera orientations (not just collinear camera centers) and gate on rotation error in `check_alignment`.
  - Make the strict `xfail` in `tests/test_fuji_colmap.py::test_window_alignment_recovers_rotation_for_collinear_cameras` pass and remove its `xfail` marker.
  - Run `.venv/bin/ruff check .` and `.venv/bin/pytest -m "not slow and not gpu" -q` (never run unfiltered full pytest in WSL2 to avoid OOM).
