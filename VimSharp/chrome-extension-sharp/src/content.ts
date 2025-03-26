// 內容腳本 - 在匹配的頁面上執行
console.log('Chrome 擴充功能內容腳本已載入');

// 這裡可以添加與頁面交互的代碼
document.addEventListener('DOMContentLoaded', () => {
  console.log('頁面已完全載入');
});

// 與背景腳本通信的示例
chrome.runtime.sendMessage({ action: 'contentScriptLoaded' }, (response) => {
  console.log('收到來自背景腳本的回應:', response);
});

// 監聽來自背景腳本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('收到消息:', message);
  sendResponse({ received: true });
  return true;
}); 