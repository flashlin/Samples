import { test, expect } from "@playwright/test";

test('download', async ({ page }) => {
   await page.goto("https://18comic.org/photo/256613?");

   await page.locator('text=我保證我已满18歲！').click();
   await page.locator('text=確定進入！').click();

   //const result = await page.evaluate(selector => document.querySelectorAll(selector) , selector);

   // const result = await page.$eval("//*[starts-with(@id, 'album_photo_') and ends-with(@id, '.jpg')]",
   //    (el: HTMLImageElement) => el.src);
   
   const result = await page.$$eval("img[src]",
      imgs => imgs.map(img => (img as HTMLImageElement).src));

   console.log("images", result);
});