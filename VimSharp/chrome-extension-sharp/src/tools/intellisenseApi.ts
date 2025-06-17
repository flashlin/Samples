import { JwtApi } from './jwtApi'

const defaultUrl = import.meta.env.VITE_INTELLISENSE_URL;

// --- Request/Response Interfaces ---
export interface AddSqlReq {
  linqSql: string
  sql: string
}

export interface GetIntellisenseListReq {
  question?: string
  user?: string
}

export interface PredictSqlContext {
  context: string
  confidence_score: number
}

export interface GetIntellisenseListResp {
  items: PredictSqlContext[]
}

export interface GetUserSqlHistoryListReq {
  user: string
}

export interface SqlHistoryRowResp {
  id: number
  linq_sql: string
  sql: string
  create_on: string | null
  create_by: string
}

export interface GetUserSqlHistoryListResp {
  items: SqlHistoryRowResp[]
}

export class IntellisenseApi {
  private _jwtApi: JwtApi;

  constructor(apiUrl: string = defaultUrl, urlPrefix: string = '/api/') {
    this._jwtApi = new JwtApi(apiUrl, urlPrefix);
  }

  public async addSql(req: AddSqlReq): Promise<{ status: string }> {
    return this._jwtApi.postAsync('addSql', req)
  }

  public async getIntellisenseList(req: GetIntellisenseListReq): Promise<GetIntellisenseListResp> {
    return this._jwtApi.postAsync('getIntellisenseList', req)
  }

  public async getUserSqlHistoryList(req: GetUserSqlHistoryListReq): Promise<GetUserSqlHistoryListResp> {
    return this._jwtApi.postAsync('getUserSqlHistoryList', req)
  }
}

export function useIntellisenseApi() {
  return new IntellisenseApi();
}