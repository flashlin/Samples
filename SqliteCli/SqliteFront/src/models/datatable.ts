export interface IDataRow {
  getValue(key: string): any;
  setValue(key: string, value: any): void;
}

export interface IDataTable {
  headers: string[];
  row: IDataRow[];
}