export interface IBannerTemplateEntity {
   id: string;
   templateContent: string;
   variablesData: string;
}

export class BannerApi 
{
   getAllTemplatesAsync(): Promise<IBannerTemplateEntity[]> {
      return this.postAsync("banner/getAllTemplates", {}); 
   }

   async postAsync(url: string, data: any){
      let resp = await fetch(`/api/${url}`,
      {
         method: "POST",
         body: JSON.stringify(data),
      });
      return resp.json();
   }
}