import { test, expect, Page } from '@playwright/test';

test('retrieve', async ({ page }) => {
   await page.goto('https://www.google.com.tw/maps/place/%E6%9F%B3%E5%B7%9D%E8%97%8D%E5%B8%B6%E6%B0%B4%E5%B2%B8/@24.1425932,120.6745042,17z/data=!3m1!4b1!4m6!3m5!1s0x34693d6d5dddf2b7:0x3b279c300f30414c!8m2!3d24.1425932!4d120.6770791!16s%2Fg%2F11c6s1bv_k?entry=ttu');
   const divTextArray = await selectorAllText(page, 'div.d4r55');
   console.log(divTextArray);
});

async function selector(page: Page, code: string) {
   return await page.evaluate(() => {
      const div = document.querySelector(code);
      return div;
   });
}

async function selectorText(page: Page, code: string) {
   const div = await selector(page, code);
   return div ? div.textContent : '';
}

async function selectorAll(page: Page, code: string) {
   return await page.evaluate(() => {
      const divs = document.querySelectorAll(code);
      debugger;
      return Array.from(divs, div => div);
   });
}

async function selectorAllText(page: Page, code: string) {
   const allDivs = await selectorAll(page, code);
   return Array.from(allDivs, div => div.textContent);
}