import * as XLSX from 'xlsx';

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
        }));
        resolve(sheets);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsArrayBuffer(file);
  });
} 