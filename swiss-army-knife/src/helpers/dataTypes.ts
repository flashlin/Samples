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