// DataTable column interface
export interface DataTableColumn {
  name: string; // column name
  type: string; // column type, e.g. 'TEXT', 'INTEGER'
}

// DataTable interface
export interface DataTable {
  tableName: string;
  columns: DataTableColumn[];
  data: any[]; // array of row objects
}

// Guess column type by value
export function guessType(value: any): string {
  if (typeof value === 'number') return 'INTEGER';
  if (typeof value === 'boolean') return 'BOOLEAN';
  if (value instanceof Date) return 'DATE';
  if (!isNaN(Date.parse(value))) return 'DATE';
  if (!isNaN(Number(value))) return 'INTEGER';
  return 'TEXT';
}
