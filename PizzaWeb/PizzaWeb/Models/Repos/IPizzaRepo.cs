using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public interface IPizzaRepo
{
    List<BannerTemplate> GetAllBannerTemplates();
    List<BannerSetting> GetBannersSetting(GetBannersSettingReq req);
    void AddBannerTemplate(AddBannerTemplateReq req);
    void AddBanner(AddBannerReq req);
    void UpdateBannerTemplate(BannerTemplate req);
    IQueryable<TemplateBannerSetting> QueryBannerJsonSettingData();
    IEnumerable<BannerSetting> QueryBannerSettings(List<TemplateBannerSetting> banners);
    List<BannerTemplateEntity> GetTemplateContents(string[] templateNames);
    void ApplyBanner(string bannerName);
    List<BannerTemplateData> GetBannersData(GetBannersDataReq req);
}