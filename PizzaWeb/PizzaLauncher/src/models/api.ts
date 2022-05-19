export interface IBannerTemplateEntity {
   id: string;
   templateContent: string;
   variablesData: string;
}

export class BannerApi 
{
   getAllTemplatesAsync(): Promise<IBannerTemplateEntity[]> {
      return this.postQueryAsync("banner/getAllTemplates", {}); 
   }

   updateTemplateAsync(req: IBannerTemplateEntity): Promise<Response> {
      return this.postAsync("banner/updateTemplate", req);
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