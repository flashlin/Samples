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
    IQueryable<TemplateBannerJsonSetting> QueryBannerJsonSettingData();
    IEnumerable<BannerSetting> QueryBannerSettings(List<TemplateBannerJsonSetting> banners);
    List<BannerTemplateEntity> GetTemplateContents(string[] templateNames);
    void ApplyBanner(string bannerName);
    List<BannerData> GetBannersData(GetBannersDataReq req);
}