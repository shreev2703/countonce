import { expect, test } from "@playwright/test";
import { existsSync, rmSync, writeFileSync } from "node:fs";

import { PAUSE_FILE } from "../../playwright.live.config";
import { fillNewJob, signIn, unique } from "../support/helpers";

/** Real API (sqlite, chunked uploads, SSE, idempotency, share, corrections) + fake worker. */

test("paused worker banner, then the worker returns and the real pipeline path completes", async ({ page }) => {
  writeFileSync(PAUSE_FILE, "");
  await signIn(page, unique("live"));
  await fillNewJob(page, { n: 6, prompt: "widgets" });
  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f]{32}$/);
  await expect(page.getByTestId("paused-banner")).toBeVisible({ timeout: 15_000 });

  rmSync(PAUSE_FILE, { force: true }); // worker comes online
  await expect(page.getByTestId("count")).toHaveText("5", { timeout: 30_000 });
  await expect(page.getByTestId("interval")).toContainText("likely 5–5");
  expect(existsSync(PAUSE_FILE)).toBe(false);
});

test("history, corrections, feedback and a share link against the real API", async ({ page, browser }) => {
  rmSync(PAUSE_FILE, { force: true });
  await signIn(page, unique("live"));
  await fillNewJob(page, { n: 6, prompt: "bolts" });
  await page.getByTestId("submit").dblclick(); // idempotent: one job
  await expect(page.getByTestId("count")).toHaveText("5", { timeout: 30_000 });

  await page.goto("/history");
  await expect(page.getByTestId("history").getByRole("listitem")).toHaveCount(1);
  await page.getByRole("link", { name: /bolts/ }).click();
  await expect(page.getByTestId("count")).toHaveText("5");

  await page.getByTestId("instance-0").getByRole("button", { name: "Not an object" }).click();
  await expect(page.getByTestId("count")).toHaveText("4");
  await page.reload(); // corrections persist on the server
  await expect(page.getByTestId("count")).toHaveText("4");

  await page.getByRole("button", { name: "Yes", exact: true }).click();
  await page.getByRole("button", { name: "Send feedback" }).click();
  await expect(page.getByTestId("feedback-thanks")).toBeVisible();

  await page.getByRole("button", { name: "Create share link" }).click();
  const link = await page.getByTestId("share-url").getByRole("link").getAttribute("href");
  const anon = await browser.newContext();
  const p = await anon.newPage();
  await p.goto(link!);
  await expect(p.getByTestId("shared-count")).toHaveText("5"); // shares show the original result
  const og = await p.locator('meta[property="og:image"]').getAttribute("content");
  const img = await anon.request.get(new URL(og!).pathname);
  expect(img.headers()["content-type"]).toBe("image/png");
  await anon.close();
});

test("another user cannot open my job", async ({ page, browser }) => {
  rmSync(PAUSE_FILE, { force: true });
  await signIn(page, unique("live"));
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\//);
  const jobUrl = page.url();
  const other = await browser.newPage();
  await signIn(other, unique("intruder"), "/history");
  const status = await other.evaluate(async (u) => (await fetch(`/api/backend/jobs/${u.split("/").pop()}`)).status, jobUrl);
  expect(status).toBe(404);
  await other.close();
});
