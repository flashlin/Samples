import createHttpClient, { HttpClient } from "@/apis/HttpClient";
import { MockAsyncMethod } from "../commons/MockUtils";

export interface ILocalQueryAppConnectOption {
  appUid: string;
  appPort: number;
}

export interface IKnockRequest {
  uniqueId: string;
  appUid: string;
}

export interface IKnockResponse {
  isSuccess: boolean;
}

export interface IGetAllTableNamesResponse {
  tableNames: string[];
}

export interface IDataRow {
  [columnName: string]: any;
}

export interface IQueryRawSqlRequest {
  sql: string;
}

export enum ColumnType {
  String,
  Number,
}

export interface ICsvHeader {
  columnType: ColumnType;
  name: string;
}

export interface ICsvRow {
  [name: string]: string;
}

export interface ICsvSheet {
  headers: ICsvHeader[];
  rows: ICsvRow[];
}

export interface IQueryRawSqlResponse {
  csvSheet: ICsvSheet;
  errorMessage: string;
}

export interface IImportLocalFileRequest {
  filePath: string;
}

export interface ILocalQueryClient {
  setConnectOption(option: ILocalQueryAppConnectOption): void;
  knockAsync(uniqueId: string): Promise<IKnockResponse>;
  getAllTableNamesAsync(): Promise<IGetAllTableNamesResponse>;
  queryRawSql(req: IQueryRawSqlRequest): Promise<IQueryRawSqlResponse>;
  importLocalFile(req: IImportLocalFileRequest): Promise<void>;
}

export class LocalQueryClient implements ILocalQueryClient {
  _httpClient!: HttpClient;
  _connectOption!: ILocalQueryAppConnectOption;

  setConnectOption(option: ILocalQueryAppConnectOption) {
    const appHost = import.meta.env.VITE_LOCAL_QUERY_APP_HOST;
    this._connectOption = option;
    this._httpClient = createHttpClient(
      `http://${appHost}:${option.appPort}/api/LocalQueryApi/`
    );
  }

  @MockAsyncMethod({ isSuccess: true })
  knockAsync(uniqueId: string): Promise<IKnockResponse> {
    const knockReq: IKnockRequest = {
      uniqueId: uniqueId,
      appUid: this._connectOption.appUid,
    };
    return this._httpClient.postAsync<IKnockResponse>("knock", knockReq);
  }

  @MockAsyncMethod({ tableNames: ["Customer", "Product"] })
  getAllTableNamesAsync(): Promise<IGetAllTableNamesResponse> {
    return this._httpClient.postAsync<IGetAllTableNamesResponse>(
      "getAllTableNames"
    );
  }

  importLocalFile(req: IImportLocalFileRequest): Promise<void> {
    return this._httpClient.postVoidAsync("importLocalFile", req);
  }

  queryRawSql(req: IQueryRawSqlRequest): Promise<IQueryRawSqlResponse> {
    return this._httpClient.postAsync<IQueryRawSqlResponse>("queryRawSql", req);
  }
}

export default function useLocalQueryClient(): ILocalQueryClient {
  return new LocalQueryClient();
}
