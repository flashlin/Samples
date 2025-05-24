import Papa from 'papaparse';

/**
 * 檢查文本是否為表格格式（每行有相同數量的分隔符）
 * @param text 要檢查的文本
 * @param delimiter 分隔符，預設為製表符 '\t'
 * @returns 是否為表格格式
 */
export function isTableFormat(text: string, delimiter: string = '\t'): boolean {
  if (!text.includes('\n')) {
    console.log(`沒有換行符`);
    return false;
  }
  
  const lines = text.split('\n').filter(line => line.trim() !== '');  
  if (lines.length === 0) { 
    console.log(`沒有任何行`);
    return false;
  }
  
  const firstLineDelimiterCount = (lines[0].match(new RegExp(delimiter, 'g')) || []).length;
  if (firstLineDelimiterCount === 0) {
    console.log(`第一行沒有分隔符`);
    return false;
  }
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    const delimiterCount = (lines[i].match(new RegExp(delimiter, 'g')) || []).length;
    if (delimiterCount !== firstLineDelimiterCount) {
      console.log(`第${i}行分隔符數量不一致`);
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
      console.log(`convert Fail`);
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

/**
 * 檢查文本是否為有效的 JSON 格式
 * @param text 要檢查的文本
 * @returns 是否為 JSON 格式
 */
export function isJsonFormat(text: string): boolean {
  if (!text || typeof text !== 'string') {
    console.log('輸入為空或不是字串');
    return false;
  }

  text = text.trim();
  
  // 檢查基本的 JSON 結構標記
  if (!(
    (text.startsWith('{') && text.endsWith('}')) || // 物件
    (text.startsWith('[') && text.endsWith(']'))    // 陣列
  )) {
    console.log('不是有效的 JSON 結構');
    return false;
  }

  try {
    // 嘗試解析 JSON
    JSON.parse(text);
    return true;
  } catch (error) {
    if (error instanceof Error) {
      console.log('JSON 解析錯誤:', error.message);
    }
    return false;
  }
}

/**
 * 將 JSON 格式的文本轉換為 CSV 格式
 * @param text 要轉換的 JSON 文本
 * @returns CSV 格式的文本，如果轉換失敗則返回原文本
 */
export function convertJsonFormatToCsv(text: string): string {
  // 檢查是否為有效的 JSON 格式
  if (!isJsonFormat(text)) {
    return text;
  }

  try {
    // 解析 JSON 文本
    const jsonData = JSON.parse(text);
    
    // 檢查是否為陣列或物件
    if (!Array.isArray(jsonData) && typeof jsonData !== 'object') {
      console.log('JSON 必須是陣列或物件格式');
      return text;
    }

    // 如果是單一物件，轉換為陣列
    const dataArray = Array.isArray(jsonData) ? jsonData : [jsonData];
    
    // 使用 Papa.unparse 將資料轉換為 CSV
    const csv = Papa.unparse(dataArray, {
      delimiter: ',',
      newline: '\n',
      header: true // 自動產生標題列
    });

    return csv;
  } catch (error) {
    console.error('JSON 轉 CSV 錯誤:', error);
    return text;
  }
}

// 添加型別定義
interface PapaParseResult {
  data: string[][];
  errors: any[];
  meta: {
    delimiter: string;
    linebreak: string;
    aborted: boolean;
    truncated: boolean;
  };
}

/**
 * 檢查文本是否為 CSV 格式（使用 Papa.parse 嘗試多種分隔符）
 * @param text 要檢查的文本
 * @returns 是否為 CSV 格式
 */
export function isCsvFormat(text: string): boolean {
  if (!text || typeof text !== 'string') {
    console.log('輸入為空或不是字串');
    return false;
  }

  // 要嘗試的分隔符列表
  const delimiters = ['\t', ',', ';'];
  
  // 檢查是否有至少一行
  const lines = text.trim().split('\n');
  if (lines.length === 0) {
    console.log('沒有任何內容');
    return false;
  }

  // 嘗試每種分隔符
  for (const delimiter of delimiters) {
    try {
      const result = Papa.parse(text, {
        delimiter: delimiter,
        preview: 5, // 只檢查前 5 行以提高效能
        skipEmptyLines: true
      }) as PapaParseResult;

      // 檢查解析結果
      if (result.data && result.data.length > 0) {
        const firstRowLength = result.data[0].length;
        
        // 確保第一行有多個欄位（至少2個）
        if (firstRowLength < 2) {
          continue;
        }

        // 檢查所有行的欄位數是否一致
        const isValid = result.data.every(row => 
          row.length === firstRowLength && 
          // 確保不是所有欄位都是空的
          row.some(cell => cell && cell.trim() !== '')
        );

        if (isValid) {
          console.log(`檢測到有效的 CSV 格式，分隔符: "${delimiter}"`);
          return true;
        }
      }
    } catch (error) {
      console.log(`使用分隔符 "${delimiter}" 解析失敗:`, error);
      continue;
    }
  }

  console.log('未檢測到有效的 CSV 格式');
  return false;
}

/**
 * 將 CSV 格式的文本轉換為 JSON 格式
 * @param text 要轉換的 CSV 文本
 * @returns JSON 格式的文本，如果轉換失敗則返回原文本
 */
export function convertCsvFormatToJson(text: string): string {
  if (!isCsvFormat(text)) {
    return text;
  }

  try {
    // 使用 Papa.parse 解析 CSV 文本，自動檢測分隔符
    const result = Papa.parse(text, {
      header: true, // 使用第一行作為標題
      skipEmptyLines: true,
      dynamicTyping: true // 自動轉換數字和布林值
    }) as PapaParseResult;

    if (!result.data || result.data.length === 0) {
      console.log('CSV 解析後沒有資料');
      return text;
    }

    // 將解析結果轉換為 JSON 字串，使用縮排格式化
    return JSON.stringify(result.data, null, 2);
  } catch (error) {
    console.error('CSV 轉 JSON 錯誤:', error);
    return text;
  }
}

/**
 * 將 CSV 格式的文本轉換為固定寬度的表格格式
 * @param text 要轉換的 CSV 文本
 * @param delimiter 輸出的分隔符
 * @returns 表格格式的文本
 */
export function convertCsvFormatToTable(text: string, delimiter: string = '\t'): string {
  if (!isCsvFormat(text)) {
    return text;
  }

  try {
    // 解析 CSV 文本
    const result = Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true
    }) as PapaParseResult;

    if (!result.data || result.data.length === 0) {
      return text;
    }

    // 獲取表頭
    const headers = Object.keys(result.data[0]);
    
    // 計算每列的最大寬度
    const columnWidths = new Map<string, number>();
    
    // 初始化寬度為表頭長度
    headers.forEach(header => {
      columnWidths.set(header, header.length);
    });

    // 計算每個欄位的最大寬度
    interface CsvRow {
      [key: string]: any;
    }

    result.data.forEach((row: CsvRow) => {
      headers.forEach(header => {
        const value = String(row[header] ?? '');
        const currentWidth = columnWidths.get(header) || 0;
        columnWidths.set(header, Math.max(currentWidth, value.length));
      });
    });

    // 生成表格內容
    const lines: string[] = [];

    // 添加表頭
    const headerLine = headers.map(header => 
      header.padEnd(columnWidths.get(header) || 0)
    ).join(delimiter);
    lines.push(headerLine);

    // 添加分隔線
    const separator = headers.map(header =>
      '-'.repeat(columnWidths.get(header) || 0)
    ).join(delimiter);
    lines.push(separator);

    // 添加數據行
    result.data.forEach((row: CsvRow) => {
      const line = headers.map(header => {
        const value = String(row[header] ?? '');
        return value.padEnd(columnWidths.get(header) || 0);
      }).join(delimiter);
      lines.push(line);
    });

    return lines.join('\n');
  } catch (error) {
    console.error('CSV 轉表格錯誤:', error);
    return text;
  }
}


export function convertCsvFormatToSql(text: string, tableName:string): string {
  if (!isCsvFormat(text)) {
    return text;
  }

  try {
    // 使用 Papa.parse 解析 CSV 文本，自動檢測分隔符
    const result = Papa.parse(text, {
      header: true, // 使用第一行作為標題
      skipEmptyLines: true,
      dynamicTyping: true // 自動轉換數字和布林值
    }) as PapaParseResult;

    if (!result.data || result.data.length === 0) {
      console.log('CSV 解析後沒有資料');
      return text;
    }

    // 取得所有欄位名稱
    const headers = Object.keys(result.data[0]);
    // 產生 INSERT 語句
    const sqlLines = result.data.map((row: any) => {
      const values = headers.map(header => {
        let value = row[header];
        if (value === null || value === undefined || value === '') {
          return 'NULL';
        } else if (typeof value === 'number' || typeof value === 'boolean') {
          return value.toString();
        } else {
          // escape 單引號
          const escaped = String(value).replace(/'/g, "''");
          return `'${escaped}'`;
        }
      });
      return `INSERT INTO ${tableName} (${headers.join(', ')}) VALUES (${values.join(', ')});`;
    });
    return sqlLines.join('\n');
  } catch (error) {
    console.error('CSV 轉 JSON 錯誤:', error);
    return text;
  }
}

export function getCsvHeadersName(csvText: string, delimiter: string = '\t'): string[] {
  // Parse CSV text to get headers
  const result = Papa.parse(csvText, {
    delimiter: delimiter,
    skipEmptyLines: true,
  });
  if (!result.data || result.data.length === 0) {
    return [];
  }
  // 取得第一行作為表頭
  const headers = result.data[0] as string[];
  if (headers.length === 1 && typeof headers[0] === 'string' && headers[0].includes(delimiter)) {
    return headers[0].split(delimiter).map(h => h.trim());
  }
  return headers.map(h => String(h).trim());
}