export interface IDataTable {
    columnNames: string[];
    rows: any[];
}

export interface IDataTableNested {
    master: IDataTable;
    detail: IDataTable[];
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

export interface IMergeTableReq {
    name: string;
    table1: IMergeTableCondition;
    table2: IMergeTableCondition;
}
