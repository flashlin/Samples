import Axios, { AxiosError, AxiosResponse } from "axios";
import { toHandlers } from "vue";
import { toastError } from "./AppToast";

export const AllIsoLangCodes = [
  "en-US",
  "zh-TW",
  "zh-CN",
  "ru-RU",
  "vi-VN",
  "es-AR"
];

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
  varType: string;
  resxName: string;
  resxList: IVariableResx[];
}

export interface IBannerSetting extends IAddBanner {
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

export interface IBannerResxEntity {
  id: number;
  isoLangCode: string;
  varType: string;
  resxName: string;
  content: string;
}

export interface IGetResxDataReq {
  resxName: string;
  varType: string;
}

export interface IUpsertResxReq {
  resxName: string;
  varType: string;
  contentList: IVariableResx[];
}

export interface IUpdateBannerVariableOption
{
    bannerName: string;
    varName: string;
    varType: string;
    resxName: string;
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
    return this.postQueryAsync("banner/getBannerSettingsPage", {
      indexPage: indexPage,
      pageSize: 10,
    });
  }

  addBannerAsync(data: IAddBanner) {
    return this.postAsync("banner/addBanner", data);
  }

  updateBannerAsync(data: IAddBanner) {
    return this.postAsync("banner/updateBannerSetting", data);
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

  getResxNamesAsync(varType: string) {
    return this.postQueryAsync<IBannerResxEntity[]>("banner/getResxNames", 
      varType
    );
  }

  getResxDataAsync(req: IGetResxDataReq): Promise<IBannerResxEntity[]> {
    return this.postQueryAsync<IBannerResxEntity[]>("banner/getResxData", req);
  }

  upsertResxAsync(req: IUpsertResxReq) {
    return this.postAsync("banner/upsertResx", req);
  }

  updateBannerVariableOptionAsync(data: IUpdateBannerVariableOption) {
    return this.postAsync("banner/updateBannerVariableOption", data);
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
