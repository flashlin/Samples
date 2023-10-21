import { test, expect, Page, Locator, ElementHandle } from '@playwright/test';
import fs from 'fs';


test('retrieve', async ({ page }) => {
   await page.goto('https://www.google.com.tw/maps/place/%E6%9F%B3%E5%B7%9D%E8%97%8D%E5%B8%B6%E6%B0%B4%E5%B2%B8/@24.1425932,120.6745042,17z/data=!3m1!4b1!4m6!3m5!1s0x34693d6d5dddf2b7:0x3b279c300f30414c!8m2!3d24.1425932!4d120.6770791!16s%2Fg%2F11c6s1bv_k?entry=ttu',
      { waitUntil: 'networkidle' });

   page.on('console', (msg) => {
      console.log(msg);
   });

   //const searchBar = await scrollBar(page);
   await page.waitForTimeout(3000);
   await scrollBar2(page);

   await clickElement(page, 'span.wNNZR', '更多評論');
   await page.waitForTimeout(1000 * 1);

   await scrollBar2(page);
   await delay(60 * 3);

   const allComments = await catchAllUserComments(page);
   let fileContext = '';
   allComments.forEach((line, index) => {
      fileContext += line.name + '\r\n';
      fileContext += line.star + '\r\n';
      fileContext += line.time + '\r\n';
      fileContext += line.comment + '\r\n';
      fileContext += "----\r\n";
   });
   await fs.writeFileSync(`output.txt`, fileContext);

   await page.pause();
});

function delay(time) {
   return new Promise((resolve) => {
      setTimeout(() => { resolve(1) }, 1000 * time)
   })
}

async function catStar(page) {
   return await page.evaluate(() => {
      const allStars = document.querySelectorAll('span.kvMYJc');
      return Array.from(allStars, a => a.getAttribute('aria-label'));
   });
   // const elements = await page.$('span.kvMYJc'); // 選取所有擁有 class="kvMYJc" 的 <span> 元素
   // const ratings: string[] = [];
   // for (const element of elements) {
   //    const ariaLabel: string = await element.getAttribute('aria-label');
   //    ratings.push(ariaLabel);
   // }
   // return ratings;
}

async function catTime(page) {
   return await selectorAllText(page, 'span.rsqaWe');
}

async function catchAllUserComments(page) {
   const allNames = await selectorAllText(page, 'div.d4r55');
   const allStars = await catStar(page);
   const allTimes = await catTime(page);
   const allComments = await selectorAllText(page, 'span.wiI7pd');
   const all = allNames.map((name, index) => ({
      name: name,
      star: allStars[index],
      time: allTimes[index],
      comment: allComments[index],
   }));
   console.log("", all)
   return all;
}

async function scrollBar2(page) {
   await page.evaluate(() => {
      const pane = document.querySelector('#pane')!.nextElementSibling!;
      const c1 = pane.children[0].children[0].children[0];
      const c2 = c1.children[1];
      const scrollbar = c2.children[0].children[0];
      scrollbar.scrollTop = scrollbar.scrollHeight;
   });
   await page.waitForTimeout(1000);
}

async function scrollBar(page) {
   const elem = await page.$('#QA0Szd > div > div > div.w6VYqd');
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