import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {

  // Go to https://18comic.org/photo/256613?
  await page.goto('https://18comic.org/photo/256613?');

  // Click text=我保證我已满18歲！
  await page.locator('text=我保證我已满18歲！').click();

  // Click text=確定進入！
  await page.locator('text=確定進入！').click();

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/256614?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/257923?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/259392?');

  // Click text=下一話 >> nth=1
  await page.locator('text=下一話').nth(1).click();
  await expect(page).toHaveURL('https://18comic.org/photo/260715?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/262158?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/263567?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/265134?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/266312?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/267656?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/269003?');

  // Click .fa.fa-angle-double-right
  await page.locator('.fa.fa-angle-double-right').click();
  await expect(page).toHaveURL('https://18comic.org/photo/270339?');

  // Click html
  await page.locator('html').click();

});