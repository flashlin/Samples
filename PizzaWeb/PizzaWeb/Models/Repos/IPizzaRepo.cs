using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public interface IPizzaRepo
{
    List<BannerTemplate> GetBannerTemplates(GetBannerTemplatesReq req);
    List<BannerSetting> GetBannersSetting(GetBannersSettingReq req);
    void AddBannerTemplate(TemplateData data);
    void AddBanner(AddBannerReq req);
    void UpdateBannerTemplate(TemplateData data);
    List<BannerTemplateEntity> GetTemplateContents(string[] templateNames);
    void ApplyBanner(string bannerName);
    List<BannerTemplateData> GetBannersData(GetBannersDataReq req);
    void DeleteBannerTemplate(string templateName);
    List<string> GetTemplateNames();
    List<BannerSetting> GetBannersSettingPage(GetBannersSettingPageReq req);
    void UpdateBannerSetting(UpdateBannerSettingReq req);
    List<BannerResxEntity> GetResxByVarType(string varType);
}