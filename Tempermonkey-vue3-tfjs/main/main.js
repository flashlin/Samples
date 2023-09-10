// ==UserScript==
// @name         utils
// @namespace    http://flash.tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       flash
// @match        *://*/*
// @require      http://localhost:5173/assets/index.js
// @resource     REMOTE_CSS http://127.0.0.1:5173/assets/index.css
// @grant        GM_xmlhttpRequest
// @grant        GM_getResourceText
// @grant        GM_addStyle
// ==/UserScript==

(function() {
   'use strict';

   // Load remote JS
   GM_xmlhttpRequest({
       method : "GET",
       // from other domain than the @match one (.org / .com):
       url : "http://localhost:5173/assets/index.js",
       onload : (ev) =>
       {
           let e = document.createElement('script');
           e.innerText = ev.responseText;
           document.head.appendChild(e);
       }
   });

   // Load remote CSS
   // @see https://github.com/Tampermonkey/tampermonkey/issues/835
   const myCss = GM_getResourceText("REMOTE_CSS");
   GM_addStyle(myCss);

   // document.querySelector('.hnname').setAttribute('data-z', true)
})();