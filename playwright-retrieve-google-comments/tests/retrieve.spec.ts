import { test, expect, Page, Locator, ElementHandle } from '@playwright/test';
import fs from 'fs';

test('retrieve', async ({ page }) => {
   await page.goto('https://www.google.com.tw/maps/place/%E6%9F%B3%E5%B7%9D%E8%97%8D%E5%B8%B6%E6%B0%B4%E5%B2%B8/@24.1425932,120.6745042,17z/data=!3m1!4b1!4m6!3m5!1s0x34693d6d5dddf2b7:0x3b279c300f30414c!8m2!3d24.1425932!4d120.6770791!16s%2Fg%2F11c6s1bv_k?entry=ttu',
      { waitUntil: 'networkidle' });

   page.on('console', (msg) => {
      console.log(msg);
   });

   const searchBar = await scrollBar(page);

   await clickElement(page, 'span.wNNZR', '更多評論');
   await page.waitForTimeout(900);
   await scroll(searchBar);

   // for(let n=0; n<10; n++) {
   //    await scroll(searchBar);
   //    await page.waitForTimeout(900);
   // }

   const allComments = await catchAllUserComments(page);
   //const fileContext = JSON.stringify(allComments);
   let fileContext = '';
   allComments.forEach((line, index) => {
      fileContext += line.name + '\r\n';
      fileContext += line.comment + '\r\n';
   });
   await fs.writeFileSync(`output.txt`, fileContext);

   await page.pause();
});

async function catchAllUserComments(page) {
   const allNames = await selectorAllText(page, 'div.d4r55');
   //console.log(allNames);
   const allComments = await selectorAllText(page, 'span.wiI7pd');
   //console.log(allComments);
   const all = allNames.map((name, index) => ({
      name: name,
      comment: allComments[index],
   }));
   //console.log(all);
   return all;
}

async function scrollBar(page) {
   const elem = await page.$('#QA0Szd > div > div > div.w6VYqd > div:nth-child(2) > div > div');
   await scroll(elem!);
   return elem;
}

async function scroll(elem: ElementHandle<SVGElement | HTMLElement>) {
   await elem!.evaluate(element => {
      element.scrollTop = element.scrollHeight;
   });
}

async function clickElement(page: Page, selectors: string, filter: string) {
   const allText = await page.evaluate(({ selectors, filter }) => {
      let elements = document.querySelectorAll(selectors);
      let elements_filtered = Array.from(elements).filter(element => {
         return element.textContent?.trim().startsWith(filter);
      });
      (elements_filtered[0] as HTMLElement).click();
   }, { selectors, filter });
   return allText;
}


async function selectorAllText(page: Page, code: string) {
   const allText = await page.evaluate((code) => {
      const divs = document.querySelectorAll(code);
      return Array.from(divs, div => div.textContent);
   }, code);
   return allText;
}

async function* iterateLocator(locator: Locator): AsyncGenerator<Locator> {
   for (let index = 0; index < await locator.count(); index++) {
      yield locator.nth(index)
   }
}