import { defineConfig, devices } from "@playwright/test";

import { webEnv } from "./playwright.config";

/**
 * Real FastAPI app + sqlite + a fake worker (`scripts/e2e_api_server.py`), driven through the
 * real Next.js BFF. Run with `npm run test:e2e:live` (needs the repo `.venv`).
 */
const PORT = Number(process.env.LIVE_WEB_PORT ?? 3200);
const API_PORT = Number(process.env.LIVE_API_PORT ?? 4020);
export const PAUSE_FILE = "/tmp/countonce-e2e-worker.pause";

export default defineConfig({
  testDir: "./e2e/live",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "on-first-retry" },
  projects: [{ name: "desktop", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `../.venv/bin/python ../scripts/e2e_api_server.py --port ${API_PORT} --pause-file ${PAUSE_FILE}`,
      url: `http://127.0.0.1:${API_PORT}/health`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { COUNTONCE_AUTH_SECRET: webEnv.COUNTONCE_AUTH_SECRET },
    },
    {
      command: `npm run build && npx next start -p ${PORT}`,
      url: `http://127.0.0.1:${PORT}`,
      reuseExistingServer: false,
      timeout: 240_000,
      env: { ...webEnv, COUNTONCE_API_URL: `http://127.0.0.1:${API_PORT}` },
    },
  ],
});
