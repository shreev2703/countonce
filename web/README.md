# CountOnce web

Next.js (App Router, TypeScript strict) + Tailwind v4 + shadcn/ui-style components +
react-three-fiber + TanStack Query. See `docs/PLAN.md` section 17.

```
npm ci
npm run dev          # http://localhost:3000
npm run lint && npm run typecheck && npm test && npm run build
npm run test:e2e     # Playwright + axe (needs: npx playwright install chromium)
```

- **API types:** `python scripts/export_openapi.py` (repo root) writes `openapi.json`; then
  `npm run gen:api` regenerates `src/lib/api/schema.gen.ts`. Commit both.
  `python scripts/export_openapi.py --check` fails when `openapi.json` is stale.
- **`result.json` schemas:** `src/lib/schemas/result.ts` (Zod, by `schema_version`, upgrades old
  versions). Tested against the backend's `tests/fixtures/schemas/v*/` fixtures.
- **Design tokens:** `src/app/globals.css` (PLAN 17.10).
