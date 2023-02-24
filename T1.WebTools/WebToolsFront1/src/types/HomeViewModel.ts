export interface ITable {
  name: string;
}

export interface ITableColumn {
  name: string;
  align?: "left" | "center" | "right";
  label: string;
  field: string | ((row: any) => any);
  sortable?: boolean;
  required?: boolean;
}

export interface IHomeViewModel {
  localFile: string;
  searchText: string;
  tableNames: string[];
  code: string;
  tabName: string;
  tableRows: any[];
  tableColumns: ITableColumn[];
}
