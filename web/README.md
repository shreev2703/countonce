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

## Signed-in flow

Routes: `/signin`, `/new`, `/jobs/[id]`, `/history`, `/s/[token]`. The browser only calls this
app; `/api/backend/*` proxies to the CountOnce API and adds the user's API key from an encrypted
httpOnly cookie. Environment variables are listed in the repo's `.env.example`.

Run locally against the real API with a fake worker (no GPU), from the repo root:

```
COUNTONCE_AUTH_SECRET=dev-secret .venv/bin/python scripts/e2e_api_server.py --port 8000
cd web && COUNTONCE_AUTH_SECRET=dev-secret COUNTONCE_SESSION_SECRET=local-session-secret-123 \
  COUNTONCE_DEV_AUTH=1 npm run dev
```

Google sign-in: create an OAuth web client with redirect URI
`<origin>/api/auth/google/callback` and set `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`.

Tests: `npm run test:e2e` (mocked API) and `npm run test:e2e:live` (real API, sqlite, fake
worker; needs the repo `.venv`).
