import { expect, type Page } from "@playwright/test";

import { photos } from "./png";

export async function signIn(page: Page, user: string, next = "/new") {
  await page.goto(`/signin?next=${encodeURIComponent(next)}`);
  await page.getByLabel("Test user name").fill(user);
  await page.getByRole("button", { name: "Sign in as test user" }).click();
  await expect(page).toHaveURL(new RegExp(`${next.replace(/\//g, "\\/")}$`));
}

/** Fill the /new form with `n` valid photos and a prompt. */
export async function fillNewJob(page: Page, opts: { n?: number; prompt?: string } = {}) {
  await page.getByTestId("file-input").setInputFiles(photos(opts.n ?? 8));
  await page.getByTestId("prompt").fill(opts.prompt ?? "red apples");
}

export const unique = (base: string) => `${base}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
