import { defineConfig, devices } from "@playwright/test";

/**
 * Default run: the signed-in flow against a mocked API (`e2e/support/mock-api.mjs`).
 * `playwright.live.config.ts` runs `e2e/live/` against the real API with a fake worker.
 */
const PORT = Number(process.env.PORT ?? 3100);
const MOCK_PORT = Number(process.env.MOCK_API_PORT ?? 4010);
export const SECRET = "e2e-secret";

export const webEnv = {
  COUNTONCE_AUTH_SECRET: SECRET,
  COUNTONCE_SESSION_SECRET: "e2e-session-secret-not-for-prod",
  COUNTONCE_DEV_AUTH: "1",
};

export default defineConfig({
  testDir: "./e2e",
  testIgnore: "**/live/**",
  fullyParallel: false,
  workers: 1, // the mock API holds shared state
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "on-first-retry" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "phone", use: { ...devices["Pixel 7"] } },
  ],
  webServer: [
    {
      command: "node e2e/support/mock-api.mjs",
      url: `http://127.0.0.1:${MOCK_PORT}/health`,
      reuseExistingServer: !process.env.CI,
      env: { MOCK_API_PORT: String(MOCK_PORT), COUNTONCE_AUTH_SECRET: SECRET },
    },
    {
      command: `npm run build && npx next start -p ${PORT}`,
      url: `http://127.0.0.1:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 240_000,
      env: { ...webEnv, COUNTONCE_API_URL: `http://127.0.0.1:${MOCK_PORT}` },
    },
  ],
});
