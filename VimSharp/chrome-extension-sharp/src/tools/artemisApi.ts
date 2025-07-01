import { JwtApi } from './jwtApi'

const defaultUrl = import.meta.env.VITE_INTELLISENSE_URL;

// DataTable 需根據實際結構定義，這裡先用 any 代替
type DataTable = any;

// LoginReq
export class LoginReq {
  env_name: string = "staging";
  username!: string;
  password!: string;
}

// LoginResp
export class LoginResp {
  message: string = "";
}

// QueryReq
export class QueryReq {
  dbFullName!: string;
  sql!: string;
}

// QueryResp
export class QueryResp {
  data: DataTable | null;
  message: string = "";
}

// GetScreenHistoryResp
export class GetScreenHistoryResp {
  history!: string[]; // base64 encoded images
}


export class ArtemisApi {
    private _jwtApi: JwtApi;
  
    constructor(apiUrl: string = defaultUrl, urlPrefix: string = '/api/artemis') {
      this._jwtApi = new JwtApi(apiUrl, urlPrefix);
    }

    // 登入
    async login(req: LoginReq): Promise<LoginResp> {
      return await this._jwtApi.postAsync<LoginResp>(`/login`, req);
    }

    // 查詢
    async query(req: QueryReq): Promise<QueryResp> {
      return await this._jwtApi.postAsync<QueryResp>(`/query`, req);
    }

    // 取得螢幕歷史（回傳 base64 image array）
    async getScreenHistory(): Promise<GetScreenHistoryResp> {
      return await this._jwtApi.postAsync<GetScreenHistoryResp>(`/getScreenHistory`, {});
    }
}