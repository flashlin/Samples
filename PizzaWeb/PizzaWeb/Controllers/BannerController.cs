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

        public void AddBannerTemplate(AddBannerTemplateReq req)
        {
            _pizzaRepo.AddBannerTemplate(req);
        }

        public void AddBanner(AddBannerReq req)
        {
           _pizzaRepo.AddBanner(req); 
        }

        public List<BannerTemplate> GetAllTemplates()
        {
            return _pizzaRepo.GetAllBannerTemplates();
        }

        public List<BannerSetting> GetBannerSettings(GetBannerSettingsReq req)
        {
            return _pizzaRepo.GetAllBanners(req);
        }

        [HttpPost]
        public void UpdateTemplate(BannerTemplate req)
        {
            _pizzaRepo.UpdateBannerTemplate(req);
        }

        public async Task<string> GetBanner(GetBannerReq req)
        {
            var allBannerData = new[]
            {
                new BannerData()
                {
                    LangCode = "TW",
                    BannerName = "A1",
                },
                new BannerData()
                {
                    LangCode = "CN",
                    BannerName = "A1",
                },
                new BannerData()
                {
                    LangCode = "EN",
                    BannerName = "A2",
                },
            };

            var bannerData = allBannerData.Where(x => x.BannerName == req.BannerName)
                .ToList();

            var bannerLogical = new BannerLogical[]
            {
            };

            var content = await _viewToStringRenderer.RenderViewToStringAsync<object>(
                @$"/banner-template:/{req.BannerName}.banner-template",
                bannerData);
            return content;
        }
    }

    public class AddBannerReq
    {
        public string BannerName { get; set; } = string.Empty;
        public string TemplateName { get; set; } = string.Empty;
        public int OrderId { get; set; }
        public Dictionary<string, TemplateVariableValue> VariablesOptions { get; set; } = new Dictionary<string, TemplateVariableValue>();
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

    public class BannerData
    {
        public string LangCode { get; set; } = "en-US";
        public string BannerName { get; set; } = string.Empty;
    }

    public class GetBannerSettingsReq
    {
        public string TemplateName { get; set; } = String.Empty;
    }
}