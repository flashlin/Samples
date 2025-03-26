// 背景腳本類型定義
interface InstallDetails {
  reason: 'install' | 'update' | 'chrome_update' | 'shared_module_update';
  previousVersion?: string;
  id?: string;
}

interface StorageData {
  [key: string]: any;
}

interface MessageResponse {
  success: boolean;
  newCount?: number;
}

// 當擴充功能安裝或更新時執行
chrome.runtime.onInstalled.addListener((details: InstallDetails) => {
  if (details.reason === 'install') {
    console.log('擴充功能已安裝');
    
    // 設置初始數據
    chrome.storage.local.set({
      installDate: new Date().toISOString(),
      counter: 0,
      showFloatingButton: true
    }, () => {
      console.log('初始數據已設置');
    });
    
    // 顯示歡迎頁面
    chrome.tabs.create({
      url: chrome.runtime.getURL('welcome.html')
    });
  } else if (details.reason === 'update') {
    console.log('擴充功能已更新到版本 ' + chrome.runtime.getManifest().version);
  }
});

// 監聽來自彈出窗口或內容腳本的消息
chrome.runtime.onMessage.addListener((message: any, sender, sendResponse: (response: MessageResponse) => void) => {
  if (message.action === 'incrementCounter') {
    // 從存儲中獲取當前計數器值
    chrome.storage.local.get('counter', (data: StorageData) => {
      const newCount = (data.counter || 0) + 1;
      
      // 更新計數器值
      chrome.storage.local.set({ counter: newCount }, () => {
        console.log('計數器已更新為: ' + newCount);
        sendResponse({ success: true, newCount: newCount });
      });
    });
    
    // 返回 true 表示將異步發送響應
    return true;
  }
  
  return false;
}); 