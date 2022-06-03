using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public interface IPizzaRepo
{
    List<BannerTemplate> GetBannerTemplates(GetBannerTemplatesReq req);
    List<BannerSetting> GetBannersSetting(GetBannersSettingReq req);
    void AddBannerTemplate(AddTemplateReq req);
    void AddBanner(AddBannerReq req);
    void UpdateBannerTemplate(BannerTemplate req);
    List<BannerTemplateEntity> GetTemplateContents(string[] templateNames);
    void ApplyBanner(string bannerName);
    List<BannerTemplateData> GetBannersData(GetBannersDataReq req);
    void DeleteBannerTemplate(string templateName);
}