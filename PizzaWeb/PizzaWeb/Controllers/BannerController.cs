using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore.Internal;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Repos;
using T1.AspNetCore;
using T1.Standard.Common;

namespace PizzaWeb.Controllers
{
    [Route("api/[controller]/[action]")]
    [ApiController]
    public class BannerController : ControllerBase
    {
        private IViewToStringRendererService _viewToStringRenderer;
        private readonly IPizzaRepo _pizzaRepo;

        public BannerController(IPizzaRepo pizzaRepo,
            IViewToStringRendererService viewToStringRenderer)
        {
            _viewToStringRenderer = viewToStringRenderer;
            _pizzaRepo = pizzaRepo;
        }

        public void AddTemplate(TemplateData data)
        {
            _pizzaRepo.AddBannerTemplate(data);
        }

        [HttpPost]
        public void UpdateTemplate(TemplateData data)
        {
            _pizzaRepo.UpdateBannerTemplate(data);
        }

        public void DeleteTemplate([FromBody] string templateName)
        {
            _pizzaRepo.DeleteBannerTemplate(templateName);
        }

        public List<BannerTemplate> GetAllTemplates(GetBannerTemplatesReq req)
        {
            return _pizzaRepo.GetBannerTemplates(req);
        }

        public void AddBanner(AddBannerReq req)
        {
            _pizzaRepo.AddBanner(req);
        }

        public void UpdateBannerVariableOption(UpdateBannerVariableOptionReq req)
        {
            _pizzaRepo.UpdateBannerVariableOption(req);
        }

        public List<string> GetTemplateNames()
        {
            return _pizzaRepo.GetTemplateNames();
        }

        public List<BannerSetting> GetBannerSettingsPage(GetBannersSettingPageReq req)
        {
            return _pizzaRepo.GetBannersSettingPage(req);
        }

        public void UpdateBannerSetting(UpdateBannerSettingReq req)
        {
            _pizzaRepo.UpdateBannerSetting(req);
        }

        public List<BannerSetting> GetBannerSettings(GetBannersSettingReq req)
        {
            return _pizzaRepo.GetBannersSetting(req);
        }

        public async Task<string> GetBanner(GetBannersDataReq req)
        {
            var bannerData = _pizzaRepo.GetBannersData(req).FirstOrDefault();

            if (bannerData == null)
            {
                return String.Empty;
            }

            var bannerLogical = new BannerLogical[]
            {
            };

            var content = await _viewToStringRenderer.RenderViewToStringAsync<object>(
                @$"/banner-template:/{bannerData.Uid}.banner-template",
                bannerData);
            return content;
        }

        public void ApplyBanner(ApplyBannerReq req)
        {
            _pizzaRepo.ApplyBanner(req.BannerName);
        }

        public List<BannerTemplateData> GetBannersData(GetBannersDataReq req)
        {
            return _pizzaRepo.GetBannersData(req);
        }

        public List<BannerResxEntity> GetResxNames([FromBody] string varType)
        {
            return _pizzaRepo.GetResxNames(varType);
        }

        public List<BannerResxEntity> GetResxData(GetResxDataReq req)
        {
            return _pizzaRepo.GetResxData(req);
        }

        public void UpsertResx(UpsertResxReq req)
        {
            _pizzaRepo.UpsertResx(req);
        }
    }

    public class UpdateBannerVariableOptionReq
    {
        public string BannerName { get; set; } = String.Empty;
        public string VarName { get; set; } = String.Empty;
        public string ResxName { get; set; } = String.Empty;
    }

    public class ResxContent
    {
        public string IsoLangCode { get; set; } = "en-US";
        public string Content { get; set; } = String.Empty;
    }

    public class UpsertResxReq
    {
        public string ResxName { get; set; } = String.Empty;
        public string VarType { get; set; } = "String";
        public List<ResxContent> ContentList { get; set; } = new List<ResxContent>();
    }

    public class GetResxDataReq
    {
        public string ResxName { get; set; } = String.Empty;
        public string VarType { get; set; } = "String";
    }

    public class UpdateBannerSettingReq
    {
        public int Id { get; set; }
        public string TemplateName { get; set; } = String.Empty;
        public string BannerName { get; set; } = String.Empty;
        public int OrderId { get; set; }
        public List<BannerVariable> Variables { get; set; } = new List<BannerVariable>();
    }

    public class GetBannersSettingPageReq
    {
        public int IndexPage { get; set; }
        public int PageSize { get; set; }
    }

    public class ApplyBannerReq
    {
        public string BannerName { get; set; } = default!;
    }

    public class GetBannersDataReq
    {
        public string BannerName { get; set; } = default!;
        public string IsoLangCode { get; set; } = "en-US";
    }

    public class AddBannerReq
    {
        public string BannerName { get; set; } = string.Empty;
        public string TemplateName { get; set; } = string.Empty;
        public int OrderId { get; set; }
        public List<VariableOption> VariablesOptions { get; set; } = new List<VariableOption>();
    }

    public class VariableResxSetting
    {
        public string VarName { get; set; } = string.Empty;
        public string VarType { get; set; } = "String";
        public string ResxName { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public string IsoLangCode { get; set; } = "en-US";
    }

    public class TemplateVariableSetting
    {
        public string VarName { get; set; } = string.Empty;
        public string VarType { get; set; } = "String";
        public string ResxName { get; set; } = string.Empty;

        public override string ToString()
        {
            return $"{VarName} {VarType}='{ResxName}'";
        }
    }

    public class BannerTemplateData
    {
        public Guid Uid { get; set; }
        public string BannerName { get; set; } = string.Empty;
        public string TemplateContent { get; set; } = string.Empty;
        public string TemplateName { get; set; } = string.Empty;
        public List<BannerData> Variables { get; set; } = new List<BannerData>();
    }

    public class BannerData
    {
        public string VarName { get; set; } = string.Empty;
        public string ResxName { get; set; } = String.Empty;
        public string Content { get; set; } = String.Empty;
    }


    public class GetBannersSettingReq
    {
        public string TemplateName { get; set; } = String.Empty;
    }
}