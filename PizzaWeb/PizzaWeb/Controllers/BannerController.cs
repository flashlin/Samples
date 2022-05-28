using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore.Internal;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using T1.AspNetCore;
using T1.Standard.Common;

namespace PizzaWeb.Controllers
{
    [Route("api/[controller]/[action]")]
    [ApiController]
    public class BannerController : ControllerBase
    {
        public PizzaDbContext _dbContext;
        private IViewToStringRendererService _viewToStringRenderer;
        private IJsonConverter _jsonConverter;
        private readonly PizzaRepo _pizzaRepo;

        public BannerController(PizzaDbContext dbContext,
            IJsonConverter jsonConverter,
            IViewToStringRendererService viewToStringRenderer)
        {
            _jsonConverter = jsonConverter;
            _viewToStringRenderer = viewToStringRenderer;
            _pizzaRepo = new PizzaRepo(dbContext);
        }

        public List<BannerTemplate> GetAllTemplates()
        {
            return _pizzaRepo.GetAllBannerTemplates();
        }

        public IEnumerable<Banner> GetBanners(GetBannersReq req)
        {
            foreach (var banner1 in _pizzaRepo.GetAllBanners())
            {
                yield return banner1;
            }
        }

        [HttpPost]
        public void UpdateTemplate(BannerTemplate req)
        {
            var bannerTemplate = _dbContext.BannerTemplates.Find(req.Id)!;
            bannerTemplate.LastModifiedTime = DateTime.Now;
            bannerTemplate.TemplateContent = req.TemplateContent;
            bannerTemplate.VariablesData = _jsonConverter.Serialize(req.Variables);
            _dbContext.BannerTemplates.Update(bannerTemplate);
            _dbContext.SaveChanges();
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

    public class VariableResxSetting
    {
        public string VarName { get; set; }
        public string VarType { get; set; }
        public string ResxName { get; set; }
        public string Content { get; set; }
        public string IsoLangCode { get; set; }
    }

    public class TemplateVariableSetting
    {
        public string VarName { get; set; }
        public string VarType { get; set; }
        public string ResxName { get; set; }
    }

    public class BannerData
    {
        public string LangCode { get; set; }
        public string BannerName { get; set; }
    }

    public class GetBannersReq
    {
        public string TemplateName { get; set; }
    }
}