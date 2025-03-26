import Papa from 'papaparse';

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

/**
 * 將表格格式的文本轉換為 CSV 格式
 * @param text 要轉換的文本
 * @param inputDelimiter 輸入文本的分隔符，預設為製表符 '\t'
 * @param outputDelimiter 輸出 CSV 的分隔符，預設為逗號 ','
 * @returns CSV 格式的文本
 */
export function convertTableFormatToCsv(
  text: string, 
  inputDelimiter: string = '\t', 
  outputDelimiter: string = ','
): string {
  if (!isTableFormat(text, inputDelimiter)) {
    return text;
  }

  try {
    // 使用 PapaParse 解析表格文本
    const result = Papa.parse(text, {
      delimiter: inputDelimiter,
      skipEmptyLines: true
    });

    if (!result.data || result.data.length === 0) {
      return text;
    }

    const csv = Papa.unparse(result.data, {
      delimiter: outputDelimiter,
      newline: '\n'
    });

    return csv;
  } catch (error) {
    console.error('CSV 轉換錯誤:', error);
    return fallbackConversion(text, inputDelimiter, outputDelimiter);
  }
}

/**
 * 備用的轉換方法，當 PapaParse 失敗時使用
 */
function fallbackConversion(
  text: string, 
  inputDelimiter: string = '\t', 
  outputDelimiter: string = ','
): string {
  const lines = text.split('\n').filter(line => line.trim() !== '');
  
  const csvLines = lines.map(line => {
    const values = line.split(inputDelimiter);
    
    // 處理每個值，如果包含逗號、引號或換行符，則用引號包裹
    const processedValues = values.map(value => {
      const trimmedValue = value.trim();
      if (
        trimmedValue.includes('"')
      ) {
        return `"${trimmedValue.replace(/"/g, '""')}"`;
      }
      return trimmedValue;
    });
    
    return processedValues.join(outputDelimiter);
  });
  
  return csvLines.join('\n');
}
