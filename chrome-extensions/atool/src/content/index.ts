// 內容腳本類型定義
interface PageInfo {
  title: string;
  url: string;
  headings: Array<{ type: string; text: string }>;
  links: number;
  images: number;
}

interface MessageResponse {
  success: boolean;
  newCount?: number;
}

// 內容腳本在網頁加載時執行
console.log('Chrome 擴充功能內容腳本已加載');

// 監聽來自背景腳本的消息
chrome.runtime.onMessage.addListener((message: any, sender, sendResponse) => {
  if (message.action === 'getPageInfo') {
    // 收集頁面信息
    const pageInfo: PageInfo = {
      title: document.title,
      url: window.location.href,
      headings: Array.from(document.querySelectorAll('h1, h2, h3')).map(h => ({
        type: (h as HTMLElement).tagName,
        text: (h as HTMLElement).textContent?.trim() || ''
      })),
      links: document.querySelectorAll('a').length,
      images: document.querySelectorAll('img').length
    };
    
    // 發送回應
    sendResponse(pageInfo);
  }
  
  return true;
});

// 創建一個浮動按鈕
function createFloatingButton(): void {
  // 檢查按鈕是否已存在
  if (document.getElementById('chrome-extension-floating-button')) {
    return;
  }
  
  // 創建按鈕元素
  const button = document.createElement('div');
  button.id = 'chrome-extension-floating-button';
  button.textContent = 'CE';
  
  // 設置按鈕樣式
  Object.assign(button.style, {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    backgroundColor: '#4285f4',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 'bold',
    cursor: 'pointer',
    zIndex: '9999',
    boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
    transition: 'transform 0.3s ease'
  });
  
  // 添加懸停效果
  button.addEventListener('mouseover', function() {
    this.style.transform = 'scale(1.1)';
  });
  
  button.addEventListener('mouseout', function() {
    this.style.transform = 'scale(1)';
  });
  
  // 添加點擊事件
  button.addEventListener('click', function() {
    // 向背景腳本發送消息
    chrome.runtime.sendMessage({ 
      action: 'incrementCounter',
      url: window.location.href
    }, (response: MessageResponse) => {
      if (response && response.success) {
        alert(`您已點擊了 ${response.newCount} 次！`);
      }
    });
  });
  
  // 將按鈕添加到頁面
  document.body.appendChild(button);
}

// 檢查是否應該顯示浮動按鈕
chrome.storage.local.get('showFloatingButton', (data: { showFloatingButton?: boolean }) => {
  if (data.showFloatingButton !== false) {
    // 默認顯示按鈕
    createFloatingButton();
  }
});

// 監聽存儲變化
chrome.storage.onChanged.addListener((changes: { [key: string]: { oldValue?: any; newValue?: any } }, namespace: string) => {
  if (namespace === 'local' && changes.showFloatingButton) {
    const showButton = changes.showFloatingButton.newValue;
    const buttonElement = document.getElementById('chrome-extension-floating-button');
    
    if (showButton && !buttonElement) {
      createFloatingButton();
    } else if (!showButton && buttonElement) {
      buttonElement?.remove();
    }
  }
}); 