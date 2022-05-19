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

   updateTemplateAsync(req: IBannerTemplateEntity): Promise<void> {
      return this.postAsync("banner/updateTemplate", req);
   }

   private async postAsync(url: string, data: any){
      let resp = await fetch(`/api/${url}`,
      {
         method: "POST",
         body: JSON.stringify(data),
      });
      return resp.json();
   }
}