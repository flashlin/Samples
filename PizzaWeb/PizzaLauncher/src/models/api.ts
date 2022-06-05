import Axios, { AxiosError, AxiosResponse } from "axios";
import { toastError } from "./AppToast";

export interface IBannerTemplateEntity {
  templateName: string;
  templateContent: string;
  variables: ITemplateVariable[];
}

export interface ITemplateVariable {
  varName: string;
  varType: string;
}

export interface ITemplateData {
  id: number;
  templateName: string;
  templateContent: string;
  variables: ITemplateVariable[];
}

export interface ITemplateVariableDict {
  [varName: string]: ITemplateVariable;
}

export interface IAddTemplateData {
  id: number;
  templateName: string;
  templateContent: string;
  variables: ITemplateVariableDict;
}

export class GetBannerReq {
  constructor(data?: Partial<GetBannerReq>) {
    Object.assign(this, data);
  }
  bannerName: string = "";
  langCode: string = "";
}

export interface IGetBannerVariablesReq {
  templateId: number;
}

export interface IBannerVariableData {
  templateId: number;
  variableName: string;
  lang: string;
  resxId: number;
  resxName: string;
  resxContent: string;
}


export interface IVariableResx {
  isoLangCode: string;
  content: string;
}

export interface IBannerVariable {
  varName: string;
  resxName: string;
  resxList: IVariableResx[];
}

export interface IBannerSetting extends IAddBanner {
  // id: number;
  // name: string;
  // orderId: number;
  // variables: IBannerVariable[];
  // templateName: string;
  lastModifiedTime: Date;
}

export interface IVariableOption {
  varName: string;
  resxName: string;
}

export interface IAddBanner {
  id: number;
  templateName: string;
  bannerName: string;
  orderId: number;
  variables: IVariableOption[];
}

export class BannerApi {
  // let apiUrl = "http://localhost:5129";
  _axios = Axios.create({
    baseURL: "",
    timeout: 1000 * 10,
  });

  getAllTemplatesAsync(indexPage: number): Promise<ITemplateData[]> {
    return this.postQueryAsync("banner/getAllTemplates", {
      indexPage: indexPage,
      pageSize: 10,
    });
  }

  addTemplateAsync(req: ITemplateData) {
    return this.postAsync("banner/addTemplate", req);
  }

  deleteTemplateAsync(templateName: string) {
     return this.postAsync("banner/deleteTemplate", templateName);
  }

  updateTemplateAsync(req: ITemplateData) {
    return this.postAsync("banner/updateTemplate", req);
  }

  getTemplateNamesAsync() {
    return this.postQueryAsync<string[]>("banner/getTemplateNames", {});
  }

  getBannerSettingsAsync(indexPage: number): Promise<IBannerSetting[]> {
    return this.postQueryAsync("banner/getBannerSettings", {
      indexPage: indexPage,
      pageSize: 10,
    });
  }

  addBanner(data: IAddBanner) {
    return this.postAsync("banner/addBanner", data);
  }

  getBannerAsync(req: GetBannerReq): Promise<string> {
    return this.postQueryAsync("banner/getBanner", req);
  }

  getBannerVariables(
    req: IGetBannerVariablesReq
  ): Promise<IBannerVariableData[]> {
    return this.postQueryAsync<IBannerVariableData[]>(
      "banner/getBannerVariables",
      req
    );
  }

  private async postAsync(url: string, data: any): Promise<AxiosResponse> {
    let apiUrl = "http://localhost:5129";
    apiUrl = "";

    try {
      const jsonData = JSON.stringify(data);
      let resp = await this._axios.post(`/api/${url}`, jsonData, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      return resp.data;
    } catch (err) {
      let errorMessage = this.getErrorMessage(err);
      toastError(errorMessage);
      throw err;
    }
  }

  getErrorMessage(error: unknown) {
    if (error instanceof Error) return error.message;
    if (error instanceof AxiosError) return error.message;
    return String(error);
  }

  private async postQueryAsync<T>(url: string, data: any) {
    let resp = await this.postAsync(url, data);
    return resp as unknown as T;
  }
}
