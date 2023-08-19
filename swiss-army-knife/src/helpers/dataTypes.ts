export interface IDataTable {
    columnNames: string[];
    rows: object[];
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