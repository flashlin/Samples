// ==UserScript==
// @name         dynamic load
// @namespace    https://github.com/flash
// @version      0.1
// @description  support load
// @author       flash
// @match        *://*/*
// @require      https://cdn.staticfile.org/jquery/3.4.1/jquery.min.js
// @updateURL    https://github.com/mygithubaccount/test/raw/master/test.user.js
// @downloadURL  https://github.com/mygithubaccount/test/raw/master/test.user.js
// @resource musicFile  https://mdn.mozillademos.org/files/2587/AudioTest%20%281%29.ogg
// @connect localhost
// @connect *
// @grant GM_setValue
// @grant GM_getValue
// @grant GM_setClipboard
// @grant GM_log
// @grant unsafeWindow
// @grant window.close
// @grant window.focus
// @grant GM_xmlhttpRequest
// @grant GM_getResourceURL
// @grant GM_getResourceText
// @grant GM_addStyle
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
    //console.log(ev.responseText);
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

    loadSource(baseUrl + indexHtml.file, (entryContent) => {
      let entryScript = document.createElement('script');
      entryScript.type = 'module';
      entryScript.innerText = entryContent;
      document.head.appendChild(entryScript);
      console.info('loaded ', baseUrl + indexHtml.file);
    });
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