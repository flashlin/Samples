export async function copyFromClipboard(
  onSuccess: (text: string) => void,
  onError: (error: string) => void
) {
  try {
    const permissionStatus = await navigator.permissions.query({ name: 'clipboard-read' as PermissionName });
    if (permissionStatus.state === 'granted' || permissionStatus.state === 'prompt') {
      const text = await navigator.clipboard.readText();
      onSuccess(text);
    } else {
      onError('沒有剪貼簿讀取權限');
      useAlternativeClipboardMethod(onSuccess, onError);
    }
  } catch (error) {
    onError('讀取剪貼簿失敗，請嘗試手動貼上');
    useAlternativeClipboardMethod(onSuccess, onError);
  }
}

export function useAlternativeClipboardMethod(
  onSuccess: (text: string) => void,
  onError: (error: string) => void
) {
  const textArea = document.createElement('textarea');
  document.body.appendChild(textArea);
  textArea.focus();
  
  try {
    const successful = document.execCommand('paste');
    if (successful) {
      onSuccess(textArea.value);
    } else {
      onError('請使用 Ctrl+V 手動貼上');
    }
  } catch (err) {
    onError('請使用 Ctrl+V 手動貼上');
  }
  
  document.body.removeChild(textArea);
}

/**
 * 將文本複製到剪貼簿
 * @param text 要複製的文本
 * @returns Promise<boolean> 複製是否成功
 */
export async function pasteToClipboard(text: string): Promise<boolean> {
  try {
    // 檢查是否支援新的 Clipboard API
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    
    // 使用傳統方法作為備選
    return useAlternativePasteMethod(text);
  } catch (error) {
    console.error('複製到剪貼簿失敗:', error);
    // 嘗試使用備選方法
    return useAlternativePasteMethod(text);
  }
}

/**
 * 使用傳統方法複製文本到剪貼簿
 * @param text 要複製的文本
 * @returns boolean 複製是否成功
 */
function useAlternativePasteMethod(text: string): boolean {
  try {
    // 創建臨時文本區域
    const textArea = document.createElement('textarea');
    textArea.value = text;
    
    // 設置樣式使其不可見
    Object.assign(textArea.style, {
      position: 'fixed',
      top: '0',
      left: '0',
      width: '2em',
      height: '2em',
      padding: '0',
      border: 'none',
      outline: 'none',
      boxShadow: 'none',
      background: 'transparent',
      opacity: '0'
    });

    // 添加到文檔中
    document.body.appendChild(textArea);
    
    // 選中文本
    textArea.focus();
    textArea.select();

    // 嘗試複製
    const successful = document.execCommand('copy');
    
    // 移除臨時元素
    document.body.removeChild(textArea);

    if (successful) {
      console.log('使用傳統方法複製成功');
      return true;
    } else {
      console.log('複製失敗');
      return false;
    }
  } catch (err) {
    console.error('傳統複製方法失敗:', err);
    return false;
  }
} 