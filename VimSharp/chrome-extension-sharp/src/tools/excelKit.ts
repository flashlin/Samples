import * as XLSX from 'xlsx';
import { DataTable, DataTableColumn, guessType } from './dataTypes';

export interface ExcelSheet {
  name: string;
  data: any[][];
}

/**
 * Read an Excel file and return all sheets' name and data.
 * @param file File object
 * @returns Promise<ExcelSheet[]>
 */
export async function getExcelFileAsync(file: File): Promise<ExcelSheet[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheets: ExcelSheet[] = workbook.SheetNames.map((name: string) => ({
          name,
          data: XLSX.utils.sheet_to_json(workbook.Sheets[name], { header: 1 })
            .filter((row: unknown) => Array.isArray(row) && row.some(cell => cell !== undefined && cell !== null && cell !== '')) as any[][]
        }));
        const filteredSheets = sheets.filter(sheet => Array.isArray(sheet.data) && !isEmptyData(sheet.data));
        resolve(filteredSheets);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsArrayBuffer(file);
  });
}

function isEmptyData(data: any[][]): boolean {
  if (!data || data.length === 0) return true;
  if( data.length === 1 && data[0].length === 0) return true;
  return false;
}

/**
 * 將 ExcelSheet 轉換為 DataTable
 * @param sheet ExcelSheet
 * @returns DataTable
 */
export function convertSheetToDataTable(sheet: ExcelSheet): DataTable {
  const columns = buildColumns(sheet.data);
  const data = buildRows(sheet.data, columns);
  return {
    tableName: sheet.name,
    columns,
    data
  };
}

// 根據第一列推斷欄位資訊
function buildColumns(data: any[][]): DataTableColumn[] {
  if (!data || data.length === 0) return [];
  const headers = data[0];
  const sampleRow = data[1] || [];
  return headers.map((header: any, idx: number) => ({
    name: String(header),
    type: guessType(sampleRow[idx])
  }));
}

// 將資料列轉為物件陣列
function buildRows(data: any[][], columns: DataTableColumn[]): any[] {
  if (!data || data.length < 2) return [];
  return data.slice(1).map(row => {
    const obj: any = {};
    columns.forEach((col, idx) => {
      obj[col.name] = row[idx];
    });
    return obj;
  });
}

/**
 * 將多個 DataTable 轉換為 XLSX workbook
 * @param dataTableList DataTable 陣列
 * @returns XLSX.WorkBook
 */
export function convertDataTableToWorkbook(dataTableList: DataTable[]): XLSX.WorkBook {
  const workbook = XLSX.utils.book_new();
  for (const dt of dataTableList) {
    // 準備 header row
    const headers = dt.columns.map(col => col.name);
    // 準備 data rows
    const dataRows = dt.data.map(rowObj => headers.map(h => rowObj[h]));
    // 合併 header 與資料
    const sheetData = [headers, ...dataRows];
    // 產生 worksheet
    const ws = XLSX.utils.aoa_to_sheet(sheetData);
    // 加入 workbook
    XLSX.utils.book_append_sheet(workbook, ws, dt.tableName);
  }
  return workbook;
} 