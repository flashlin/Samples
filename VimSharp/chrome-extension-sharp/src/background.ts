// 背景腳本 - 在擴充功能的生命週期內持續運行
console.log('Chrome 擴充功能背景腳本已啟動');

// 監聽來自內容腳本的消息
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  console.log('收到來自內容腳本的消息:', message);
  
  if (message.action === 'contentScriptLoaded') {
    sendResponse({ status: 'background received your message' });
  }
  
  return true; // 保持消息通道開啟，以便異步回應
});

// 監聽標籤頁更新事件
chrome.tabs.onUpdated.addListener((_tabId: number, changeInfo: chrome.tabs.TabChangeInfo, tab: chrome.tabs.Tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    console.log('標籤頁已完成載入:', tab.url);
  }
}); 