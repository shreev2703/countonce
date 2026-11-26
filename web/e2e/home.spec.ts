import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("home renders and has no serious accessibility violations", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("without counting it twice");
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
});

test("proof points show measured numbers or an empty state, and the example link works", async ({ page }) => {
  await page.goto("/");
  // Fuji numbers come from the committed benchmark summary; phone photos have none yet.
  await expect(page.getByText("40.1 for CountOnce v2")).toBeVisible();
  await expect(page.getByText("No published number yet")).toHaveCount(1);
  await expect(page.getByRole("link", { name: "Count your own photos" })).toHaveAttribute("href", "/new");
  await page.getByRole("link", { name: "Try a live example" }).click();
  await expect(page).toHaveURL(/\/demo\/tiny$/);
});

test("demo gallery lists the exported scenes", async ({ page }) => {
  await page.goto("/demo");
  await expect(page.getByRole("heading", { level: 1, name: "Demo gallery" })).toBeVisible();
  await expect(page.getByRole("link", { name: "synthetic test scene" })).toBeVisible();
});
