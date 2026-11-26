import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const BACKEND = /\/(api|jobs)\//;

test.describe("results experience on the tiny demo (no backend)", () => {
  test.beforeEach(async ({ page }) => {
    // The demo must work with no backend: fail the test if any backend-looking request is made.
    page.on("request", (r) => expect(new URL(r.url()).pathname).not.toMatch(BACKEND));
    await page.goto("/demo/tiny");
    await expect(page.getByTestId("count")).toHaveText("5");
  });

  test("count card, honesty line and viewer render", async ({ page }) => {
    await expect(page.getByTestId("interval")).toContainText("likely 5–25");
    await expect(page.getByTestId("confidence")).toContainText("confidence");
    await expect(page.getByTestId("honesty")).toContainText("Precomputed example");
    await expect(page.getByTestId("honesty")).toContainText("synthetic test scene");
    // Phones wait for the button; desktop loads the viewer once the page is idle.
    const load = page.getByRole("button", { name: "Load 3D view" });
    if (await load.isVisible()) await load.click();
    await expect(page.getByTestId("viewer").locator("canvas")).toBeVisible({ timeout: 15_000 });
  });

  test("selecting an object highlights its boxes in every photo (two-way linking)", async ({ page }) => {
    await page.getByTestId("object-2").click();
    await expect(page.getByTestId("viewer")).toHaveAttribute("data-selected", "2");
    const hot = page.locator('[data-testid^="box-"][data-hot="true"]');
    await expect(hot).toHaveCount(5); // one box per photo
    // hovering a box in a photo highlights its object chip
    await page.getByTestId("box-0:0").dispatchEvent("pointerover");
    await expect(page.getByTestId("object-0")).toHaveAttribute("aria-pressed", "false");
    await expect(page.getByTestId("object-0")).toHaveClass(/bg-accent/);
  });

  test("clicking a photo box selects its object", async ({ page }) => {
    await page.getByTestId("box-1:1").click();
    await expect(page.getByTestId("viewer")).toHaveAttribute("data-selected", "0");
  });

  test("photo strip flies to the camera and fades the photo in", async ({ page }) => {
    await page.getByTestId("strip-3").click();
    await expect(page.getByTestId("viewer")).toHaveAttribute("data-camera", "3");
    await expect(page.getByTestId("camera-photo")).toBeVisible();
    await page.getByTestId("viewer").press("Escape");
    await expect(page.getByTestId("camera-photo")).toHaveCount(0);
  });

  test("keyboard: arrow keys step through objects", async ({ page }) => {
    const viewer = page.getByTestId("viewer");
    await viewer.focus();
    await viewer.press("ArrowRight");
    await expect(viewer).toHaveAttribute("data-selected", "0");
    await viewer.press("ArrowRight");
    await expect(viewer).toHaveAttribute("data-selected", "1");
    await viewer.press("ArrowLeft");
    await viewer.press("ArrowLeft");
    await expect(viewer).toHaveAttribute("data-selected", "4"); // wraps
  });

  test("tabs: Photos, Explain, Warnings, Details", async ({ page }) => {
    await expect(page.getByRole("tab", { name: "Photos" })).toHaveAttribute("aria-selected", "true");
    await page.getByRole("tab", { name: "Explain" }).click();
    await page.getByRole("group", { name: "Objects", exact: true }).getByRole("button", { name: "#1", exact: true }).click();
    await expect(page.getByTestId("explanation")).toContainText("same physical object");
    await page.getByRole("tab", { name: /Warnings/ }).click();
    await expect(page.getByTestId("warning")).toHaveCount(2);
    await page.getByRole("tab", { name: "Details" }).click();
    await expect(page.getByText("color_blob@1")).toBeVisible();
    await page.getByRole("tab", { name: "Details" }).press("ArrowLeft");
    await expect(page.getByRole("tab", { name: /Warnings/ })).toHaveAttribute("aria-selected", "true");
  });

  test("photo status filter", async ({ page }) => {
    await page.getByRole("button", { name: "new", exact: true }).click();
    await expect(page.locator('[data-testid^="box-"][data-status="duplicate"]')).toHaveCount(0);
    await expect(page.locator('[data-testid^="box-"][data-status="new"]')).toHaveCount(5);
  });

  test("no serious accessibility violations", async ({ page }) => {
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
      .analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });

  test("phone layout stacks the count card above the viewer without horizontal scroll", async ({ page }, info) => {
    test.skip(info.project.name !== "phone");
    const top = (testid: string) =>
      page.getByTestId(testid).evaluate((el) => el.getBoundingClientRect().top + window.scrollY);
    const cardTop = await top("count");
    const viewerTop = await top("viewer");
    const viewerH = await page.getByTestId("viewer").evaluate((el) => el.getBoundingClientRect().height);
    const tabsTop = await page.getByRole("tablist").evaluate((el) => el.getBoundingClientRect().top + window.scrollY);
    expect(cardTop).toBeLessThan(viewerTop);
    expect(tabsTop).toBeGreaterThan(viewerTop + viewerH - 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  });
});

test("without WebGL the count and photos still show", async ({ page }) => {
  await page.addInitScript(() => {
    const real = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...args: unknown[]) {
      return type.startsWith("webgl") ? null : (real as (...a: unknown[]) => unknown).call(this, type, ...args);
    } as typeof real;
  });
  await page.goto("/demo/tiny");
  await expect(page.getByTestId("count")).toHaveText("5");
  const load = page.getByRole("button", { name: "Load 3D view" });
  if (await load.isVisible()) await load.click();
  await expect(page.getByText("The 3D view could not start on this device")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("tab", { name: "Photos" })).toBeVisible();
});

test("phones load no point cloud until asked", async ({ page, isMobile }) => {
  test.skip(!isMobile, "desktop loads the viewer once the page is idle");
  const requested: string[] = [];
  page.on("request", (r) => requested.push(r.url()));
  await page.goto("/demo/tiny", { waitUntil: "networkidle" });
  await expect(page.getByTestId("viewer-poster")).toBeVisible();
  expect(requested.filter((u) => u.endsWith("points.bin"))).toEqual([]);
  await page.getByRole("button", { name: "Load 3D view" }).click();
  await expect(page.getByTestId("viewer").locator("canvas")).toBeVisible({ timeout: 15_000 });
  expect(requested.some((u) => u.endsWith("points.bin"))).toBe(true);
});
