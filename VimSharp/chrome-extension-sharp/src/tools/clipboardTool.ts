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