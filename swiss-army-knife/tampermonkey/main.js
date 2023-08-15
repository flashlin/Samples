// ==UserScript==
// @name         dynamic load
// @namespace    https://github.com/flash
// @version      0.1
// @description  support load
// @author       flash
// @match        *://*/*
// @require      https://cdn.staticfile.org/jquery/3.4.1/jquery.min.js
// @resource musicFile  https://mdn.mozillademos.org/files/2587/AudioTest%20%281%29.ogg
// @connect localhost
// @connect *
// @grant GM_setValue
// @grant GM_getValue
// @grant GM_setClipboard
// @grant GM_log
// @grant GM_xmlhttpRequest
// @grant unsafeWindow
// @grant window.close
// @grant window.focus
// @grant GM_getResourceURL
// ==/UserScript==
function loadSource(url, callback) {
  GM_xmlhttpRequest({
    method: 'GET',
    url: url,
    onload: (ev) => {
      callback(ev.responseText);
    },
  });
}

const baseUrl = 'http://localhost:8002/';
GM_xmlhttpRequest({
  method: 'GET',
  url: 'http://localhost:8002/manifest.json',
  onload: (ev) => {
    console.log(ev.responseText);
    const data = JSON.parse(ev.responseText);
    const indexHtml = data['index.html'];
    indexHtml.css.forEach((cssUrl) => {
      loadSource(baseUrl + cssUrl, (linkContent) => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.innerHTML = linkContent;
        document.head.appendChild(link);
      });
    });
    //let e = document.createElement('script');
    //e.innerText = ev.responseText;
    //document.head.appendChild(e);
  },
});

// const baseUrl = 'http://localhost:8002/';
// fetch(`${baseUrl}manifest.json`)
//   .then((response) => response.json())
//   .then((data) => {
//     const indexHtml = data['index.html'];
//     indexHtml.css.forEach((cssUrl) => {
//       const link = document.createElement('link');
//       link.rel = 'stylesheet';
//       link.href = baseUrl + cssUrl;
//       document.head.appendChild(link);
//     });
//     const script = document.createElement('script');
//     script.src = baseUrl + indexHtml.file;
//     document.body.appendChild(script);
//   })
//   .catch((error) => {
//     console.error('Error downloading or parsing manifest.json:', error);
//   });
