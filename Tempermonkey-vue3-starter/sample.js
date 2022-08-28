// ==UserScript==
// @name         修改Vue3
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://bbs.tampermonkey.net.cn/forum.php
// @icon         https://www.google.com/s2/favicons?domain=tampermonkey.net.cn
// @require      https://raw.githubusercontent.com/lihengdao666/Modify-Tampermonkey-Libs/master/vue.js
// @grant        unsafeWindow
// @grant      GM_getResourceText
// @grant      GM_addStyle
// ==/UserScript==
let text = `<div id="app" style="position: absolute;top: 50vh;left: 50vw;background: #fb7d7d;width: 100px;height: 100px;">
           {{ message }}
    </div>`

var el = document.createElement('div')
el.innerHTML = text;
document.body.append(el)
const App = {
   data() {
      return {
         message: "Hello Element Plus",
      };
   },
};
const app = Vue.createApp(App);

app.mount("#app");