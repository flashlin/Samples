chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "logFormData") {
        console.log("Form Data Submitted:");
        for (let key in request.formData) {
            console.log(key + ": " + request.formData[key]);
        }
    }
});

// 監聽表單提交事件
document.addEventListener('submit', function (e) {
    const form = e.target;
    const formData = new FormData(form);
    console.log("Form Submitted:");
    for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
    }
});