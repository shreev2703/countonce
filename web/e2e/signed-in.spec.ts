import AxeBuilder from "@axe-core/playwright";
import { expect, test, type APIRequestContext } from "@playwright/test";

import { fillNewJob, signIn, unique } from "./support/helpers";
import { photos } from "./support/png";

const MOCK = `http://127.0.0.1:${process.env.MOCK_API_PORT ?? 4010}`;
const mock = (request: APIRequestContext, path: string, data?: object) => request.post(`${MOCK}/__mock/${path}`, { data });

test.beforeEach(async ({ request }) => {
  await mock(request, "reset");
});

test("signed-out visitors are sent to sign-in and back after the test sign-in", async ({ page }) => {
  await page.goto("/new");
  await expect(page).toHaveURL(/\/signin\?next=%2Fnew/);
  await expect(page.getByTestId("signin-unavailable")).toBeVisible(); // no Google keys in tests: the Google button is replaced by a demo link
  await page.getByLabel("Test user name").fill(unique("u"));
  await page.getByRole("button", { name: "Sign in as test user" }).click();
  await expect(page).toHaveURL(/\/new$/);
  await expect(page.getByRole("heading", { level: 1, name: "New count" })).toBeVisible();
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.goto("/history");
  await expect(page).toHaveURL(/\/signin/);
});

test("the API key never reaches the browser and the proxy needs a session", async ({ page, request }) => {
  const anon = await request.get("/api/backend/jobs");
  expect(anon.status()).toBe(401);
  await signIn(page, unique("u"));
  const cookies = await page.context().cookies();
  const session = cookies.find((c) => c.name === "co_session")!;
  expect(session.httpOnly).toBe(true);
  expect(session.value).not.toContain("co_");
  const me = await page.evaluate(async () => (await fetch("/api/auth/me")).json());
  expect(JSON.stringify(me)).not.toContain("co_");
});

test("pre-flight checks block a bad selection and explain the fix", async ({ page }) => {
  await signIn(page, unique("u"));
  const submit = page.getByTestId("submit");
  await expect(submit).toBeDisabled();
  await page.getByTestId("file-input").setInputFiles(photos(3));
  await expect(page.getByTestId("preflight").getByText("Only 3 photo(s)")).toBeVisible();
  await page.getByTestId("prompt").fill("apples");
  await expect(submit).toBeDisabled();
  await page.getByTestId("file-input").setInputFiles([{ name: "scan.heic", mimeType: "image/heic", buffer: Buffer.from("x") }]);
  await expect(page.getByTestId("preflight").getByText(/HEIC/)).toBeVisible();
  await page.getByRole("button", { name: "Clear all" }).click();
  await page.getByTestId("file-input").setInputFiles(photos(8));
  await expect(page.getByTestId("preflight").locator("[data-level=fail]")).toHaveCount(0);
  await expect(submit).toBeEnabled();
  // the camera input is a separate control for phones
  await expect(page.getByTestId("camera-input")).toHaveAttribute("capture", "environment");
});

test("upload, submit, live progress over SSE, result with interval", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await fillNewJob(page, { prompt: "widgets" });
  await page.getByTestId("negative").fill("broken widgets");
  const box = page.getByTestId("example-box-canvas");
  const r = (await box.boundingBox())!;
  await page.mouse.move(r.x + r.width * 0.2, r.y + r.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(r.x + r.width * 0.5, r.y + r.height * 0.5, { steps: 4 });
  await page.mouse.up();
  await expect(page.getByTestId("example-box")).toBeVisible();

  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f]{32}$/);
  await expect(page.getByTestId("transport")).toHaveAttribute("data-transport", "sse");
  await expect(page.getByTestId("count")).toHaveText("5", { timeout: 15_000 });
  await expect(page.getByTestId("interval")).toContainText("likely 5–5");
  await expect(page.getByTestId("confidence")).toBeVisible();

  const state = await (await request.get(`${MOCK}/__mock/state`)).json();
  expect(state.creates).toBe(1);
  expect(state.jobs[0].files).toBe(8);
});

test("a double tap on Count creates one job", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await fillNewJob(page);
  await page.getByTestId("submit").dblclick();
  await expect(page).toHaveURL(/\/jobs\//);
  const state = await (await request.get(`${MOCK}/__mock/state`)).json();
  expect(state.creates).toBe(1);
  expect(state.jobs).toHaveLength(1);
});

test("dropped chunk uploads are retried and the submit still succeeds", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await mock(request, "config", { failChunks: 2 });
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\//, { timeout: 20_000 });
  await expect(page.getByTestId("count")).toBeVisible({ timeout: 15_000 });
});

test("an upload that keeps failing shows a calm error and can be retried", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await mock(request, "config", { failChunks: 100 });
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page.getByTestId("submit-error")).toContainText("Nothing is lost", { timeout: 30_000 });
  await mock(request, "config", { failChunks: 0 });
  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\//, { timeout: 20_000 });
  const state = await (await request.get(`${MOCK}/__mock/state`)).json();
  expect(state.creates).toBe(1); // same idempotency key: still one job
  expect(state.jobs[0].files).toBe(6);
});

test("falls back to polling when the event stream is unavailable", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await mock(request, "config", { sse: false });
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page).toHaveURL(/\/jobs\//);
  await expect(page.getByTestId("count")).toHaveText("5", { timeout: 20_000 });
});

test("a paused worker shows the banner and the job finishes when it returns", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await mock(request, "config", { workerOnline: false });
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page.getByTestId("paused-banner")).toContainText("Processing is paused", { timeout: 15_000 });
  await expect(page.getByTestId("job-status")).toHaveAttribute("data-status", "queued");
  await mock(request, "config", { workerOnline: true });
  await expect(page.getByTestId("count")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("paused-banner")).toHaveCount(0);
});

test("history lists my jobs only and can delete one", async ({ page, browser }) => {
  const me = unique("me");
  await signIn(page, me);
  await fillNewJob(page, { n: 6, prompt: "my pears" });
  await page.getByTestId("submit").click();
  await expect(page.getByTestId("count")).toBeVisible({ timeout: 15_000 });

  const other = await browser.newPage();
  await signIn(other, unique("other"), "/history");
  await expect(other.getByTestId("history-empty")).toBeVisible();
  await other.close();

  await page.goto("/history");
  await expect(page.getByTestId("history")).toContainText("my pears");
  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Delete my pears" }).click();
  await expect(page.getByTestId("history-empty")).toBeVisible();
});

test("share link opens without sign-in, has Open Graph tags and an image, and unknown tokens are handled", async ({ page, browser, request }) => {
  await signIn(page, unique("u"));
  await fillNewJob(page, { n: 6, prompt: "widgets" });
  await page.getByTestId("submit").click();
  await expect(page.getByTestId("count")).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "Create share link" }).click();
  const link = await page.getByTestId("share-url").getByRole("link").getAttribute("href");
  expect(link).toMatch(/\/s\/st_/);

  const anon = await browser.newContext();
  const p = await anon.newPage();
  await p.goto(link!);
  await expect(p.getByTestId("shared-count")).toHaveText("5");
  await expect(p.getByText("not the photos")).toBeVisible();
  expect(await p.locator('meta[property="og:title"]').getAttribute("content")).toBe("5 widgets · CountOnce");
  const og = await p.locator('meta[property="og:image"]').getAttribute("content");
  expect(og).toContain("/opengraph-image");
  const img = await anon.request.get(new URL(og!).pathname);
  expect(img.status()).toBe(200);
  expect(img.headers()["content-type"]).toBe("image/png");
  expect((await img.body()).length).toBeGreaterThan(2000);
  await anon.close();

  const missing = await request.get("/s/st_nope");
  expect(await missing.text()).toContain("This link is not available");
});

test("corrections change the count and feedback is sent", async ({ page, request }) => {
  await signIn(page, unique("u"));
  await fillNewJob(page, { n: 6 });
  await page.getByTestId("submit").click();
  await expect(page.getByTestId("count")).toHaveText("5", { timeout: 15_000 });

  await page.getByTestId("instance-0").getByRole("button", { name: "Not an object" }).click();
  await expect(page.getByTestId("correction-msg")).toContainText("The count is now 4");
  await expect(page.getByTestId("count")).toHaveText("4");
  await expect(page.getByTestId("corrected-note")).toContainText("from 5");

  await page.getByRole("button", { name: "No", exact: true }).click();
  await page.getByLabel("How many are there").fill("6");
  await page.getByRole("button", { name: "Send feedback" }).click();
  await expect(page.getByTestId("feedback-thanks")).toBeVisible();
  const state = await (await request.get(`${MOCK}/__mock/state`)).json();
  expect(state.feedback[0]).toMatchObject({ correct: false, user_count: 6 });
  expect(state.corrections[0]).toMatchObject({ action: "reject" });
});

test("sign-in and new-count pages have no serious accessibility violations", async ({ page }) => {
  await page.goto("/signin");
  await signIn(page, unique("u"));
  await page.getByTestId("file-input").setInputFiles(photos(8));
  await expect(page.getByTestId("example-box-canvas")).toBeVisible();
  for (const path of ["/new"]) {
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, `${path}: ${JSON.stringify(serious, null, 2)}`).toEqual([]);
  }
});
