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
