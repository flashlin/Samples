export interface IDataTable {
    columnNames: string[];
    rows: any[];
}

export interface IMasterDetailDataTable {
    master: IDataTable;
    detail: IDataTable[];
}

export interface IColumnOption {
    isSelected: boolean;
}

export interface IDisplayDataTable extends IDataTable {
    columnOptions: IColumnOption[];
}

export interface IListItem {
    label: string;
    value: any;
}

export enum MessageTypes {
    Success = "success",
    Error = "error",
    Warning = "warning",
}

export type MessageType = MessageTypes.Success | MessageTypes.Error | MessageTypes.Warning;

export interface IMergeTableCondition {
    name: string;
    columns: string[];
    joinOnColumns: string[];
}

export interface IMergeTableForm {
    name: string;
    table1: IMergeTableCondition;
    table2: IMergeTableCondition;
}
