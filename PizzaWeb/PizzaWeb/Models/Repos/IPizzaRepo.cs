using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public interface IPizzaRepo
{
    List<BannerTemplate> GetBannerTemplates(GetBannerTemplatesReq req);
    List<BannerSetting> GetBannersSetting(GetBannersSettingReq req);
    void AddBannerTemplate(UpdateTemplateData data);
    void AddBanner(AddBannerReq req);
    void UpdateBannerTemplate(UpdateTemplateData data);
    List<BannerTemplateEntity> GetTemplateContents(string[] templateNames);
    void ApplyBanner(string bannerName);
    List<BannerTemplateData> GetBannersData(GetBannersDataReq req);
    void DeleteBannerTemplate(string templateName);
}