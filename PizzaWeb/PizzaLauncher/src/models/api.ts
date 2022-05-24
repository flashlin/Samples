export interface IBannerTemplateEntity {
   id: string;
   templateContent: string;
   variablesData: string;
}

export interface ITemplateVariable 
{
   name: string;
   fulltype: string;
}

export interface IBannerTemplateData {
   id: string;
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

export class BannerApi 
{
   getAllTemplatesAsync(): Promise<IBannerTemplateData[]> {
      return this.postQueryAsync("banner/getAllTemplates", {}); 
   }

   updateTemplateAsync(req: IBannerTemplateData): Promise<Response> {
      return this.postAsync("banner/updateTemplate", req);
   }

   getBannerAsync(req: GetBannerReq): Promise<string> {
      return this.postQueryAsync("banner/getBanner", req);
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

   private async postQueryAsync(url: string, data: any){
      let resp = await this.postAsync(url, data);
      return resp.json();
   }
}