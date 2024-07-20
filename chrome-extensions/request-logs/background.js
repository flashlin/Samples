// chrome.declarativeNetRequest.updateDynamicRules({
//     addRules: [{
//         'id': 1001,
//         'priority': 1,
//         'action': {
//             'type': 'block'
//         },
//         'condition': {
//             'urlFilter': 'google.com',
//             'resourceTypes': [
//                 'csp_report', 'font', 'image', 'main_frame', 'media', 'object', 'other', 'ping', 'script',
//                 'stylesheet', 'sub_frame', 'webbundle', 'websocket', 'webtransport', 'xmlhttprequest'
//             ]
//         }
//     }],
//     removeRuleIds: [1001]
// })


function getUrlParameters(url) {
    // 解析 URL 參數
    let urlParams = new URLSearchParams(new URL(url).search);
    return Object.fromEntries(urlParams);
}


chrome.webRequest.onBeforeRequest.addListener(
    function (details) {
        if (details.method === "POST" && details.type === "main_frame") {
            chrome.tabs.sendMessage(details.tabId, {
                action: "logFormData",
                formData: details.requestBody.formData
            });
            return { cancel: false };
        }

        if (details.method === "POST" || details.method === "GET") {
            console.log("URL:", details.url);
            console.log("Method:", details.method);

            // 解析 URL 參數
            let urlParams = getUrlParameters(details.url);
            console.log("URL Parameters:", urlParams);

            // 解析 POST 數據
            if (details.method === "POST" && details.requestBody) {
                if (details.requestBody.raw) {
                    // 原始數據
                    let postedString = decodeURIComponent(String.fromCharCode.apply(null,
                        new Uint8Array(details.requestBody.raw[0].bytes)));
                    console.log("POST data:", postedString);
                } else if (details.requestBody.formData) {
                    // 表單數據
                    console.log("Form data:", details.requestBody.formData);
                }
            }
        }
        return { cancel: false };
    },
    { urls: ["<all_urls>"] },
    ["requestBody"]
);