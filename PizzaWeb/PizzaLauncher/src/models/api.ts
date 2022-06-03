export interface IBannerTemplateEntity {
   templateName: string;
   templateContent: string;
   variables: ITemplateVariable[];
}

export interface ITemplateVariable 
{
   varName: string;
   varType: string;
}

export interface ITemplateData {
   id: number;
   templateName: string;
   templateContent: string;
   variables: ITemplateVariable[];
}

export class GetBannerReq {
   constructor(data?: Partial<GetBannerReq>)
   {
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

export class BannerApi 
{
   getAllTemplatesAsync(indexPage: number): Promise<ITemplateData[]> {
      return this.postQueryAsync("banner/getAllTemplates", {
         indexPage: indexPage,
         pageSize: 10,
      }); 
   }

   addTemplateAsync(req: ITemplateData): Promise<Response> {
      return this.postAsync("banner/addTemplate", req);
   }

   updateTemplateAsync(req: ITemplateData): Promise<Response> {
      return this.postAsync("banner/updateTemplate", req);
   }

   getBannerAsync(req: GetBannerReq): Promise<string> {
      return this.postQueryAsync("banner/getBanner", req);
   }

   getBannerVariables(req: IGetBannerVariablesReq): Promise<IBannerVariableData[]> {
      return this.postQueryAsync<IBannerVariableData[]>("banner/getBannerVariables", req);
   }

   private async postAsync(url: string, data: any){
      let apiUrl = "http://localhost:5129";
      apiUrl = "";
      let resp = await fetch(`${apiUrl}/api/${url}`,
      {
         headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
         },
         method: "POST",
         body: JSON.stringify(data),
      });
      return resp;
   }

   private async postQueryAsync<T>(url: string, data: any){
      let resp = await this.postAsync(url, data);
      return resp.json() as unknown as T;
   }
}