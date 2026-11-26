import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = [
  ["/benchmark", "Benchmark"],
  ["/how-it-works", "How it works"],
  ["/status", "Status"],
  ["/about", "About"],
  ["/privacy", "Privacy"],
] as const;

for (const [path, heading] of PAGES) {
  test(`${path} renders without placeholders or serious accessibility violations`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    await expect(page.locator("body")).not.toContainText("TODO");
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });
}

test("how-it-works steps through to the count", async ({ page }) => {
  await page.goto("/how-it-works");
  for (let i = 0; i < 3; i++) await page.getByRole("button", { name: "Next stage" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Count" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Explore this scene in 3D" })).toHaveAttribute("href", "/demo/tiny");
});

test("status shows the worker state without signing in", async ({ page, request }) => {
  await request.post("http://127.0.0.1:4010/__mock/reset");
  await page.goto("/status");
  await expect(page.getByTestId("status-panel")).toContainText("Online");
  await expect(page.getByText("No finished jobs yet.")).toBeVisible();
});

test("benchmark shows measured tables or the empty state", async ({ page }) => {
  await page.goto("/benchmark");
  // results/ is local-only apart from the committed summary, so either state is valid.
  await expect(page.getByTestId("benchmark-empty").first().or(page.getByRole("region", { name: "Benchmark values" }))).toBeVisible();
});
