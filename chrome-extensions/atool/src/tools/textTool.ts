/**
 * 檢查文本是否為表格格式（每行有相同數量的分隔符）
 * @param text 要檢查的文本
 * @param delimiter 分隔符，預設為製表符 '\t'
 * @returns 是否為表格格式
 */
export function isTableFormat(text: string, delimiter: string = '\t'): boolean {
  if (!text.includes('\n')) {
    return false;
  }
  
  const lines = text.split('\n').filter(line => line.trim() !== '');
  
  if (lines.length === 0) {
    return false;
  }
  
  const firstLineDelimiterCount = (lines[0].match(new RegExp(delimiter, 'g')) || []).length;
  if (firstLineDelimiterCount === 0) {
    return false;
  }
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    const delimiterCount = (lines[i].match(new RegExp(delimiter, 'g')) || []).length;
    if (delimiterCount !== firstLineDelimiterCount) {
      return false;
    }
  }
  
  return true;
}
