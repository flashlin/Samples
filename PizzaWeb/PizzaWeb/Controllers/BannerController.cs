using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
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
        private PizzaDbContext _dbContext;
        private IViewToStringRendererService _viewToStringRenderer;
        private IJsonConverter _jsonConverter;

        public BannerController(PizzaDbContext dbContext,
            IJsonConverter jsonConverter,
            IViewToStringRendererService viewToStringRenderer)
        {
            _jsonConverter = jsonConverter;
            _viewToStringRenderer = viewToStringRenderer;
            _dbContext = dbContext;
        }

        public List<BannerTemplate> GetAllTemplates()
        {
            var bannerTemplates = _dbContext.BannerTemplates
                .AsNoTracking()
                .ToList();
            return bannerTemplates.Select(x => BannerTemplate.From(x)).ToList();
        }

        public IEnumerable<Banner> GetBanners(GetBannersReq req)
        {
            var banners = GetBannersData();
            foreach (var banner in banners)
            {
                var q2 = GetTemplateVariablesSettings(banner);
                var q3 = GetAllVariableSettings(q2);
                var q4 = GetBannerVariables(q3);
                yield return new Banner
                {
                    Id = banner.Id,
                    Name = banner.Name,
                    TemplateName = banner.TemplateName,
                    OrderId = banner.OrderId,
                    Variables = q4.ToList()
                };
            }
        }

        private static List<BannerVariable> GetBannerVariables(List<VariableResxSetting> q3)
        {
            var q4 = (from tb1 in q3
                group tb1 by new {tb1.VarName, tb1.ResxName}
                into g1
                select new BannerVariable
                {
                    VarName = g1.Key.VarName,
                    ResxName = g1.Key.ResxName,
                    ResxList = g1.Select(x => new VariableResx
                    {
                        IsoLangCode = x.IsoLangCode,
                        Content = x.Content,
                    }).ToList()
                }).ToList();
            return q4;
        }

        private List<VariableResxSetting> GetAllVariableSettings(List<TemplateVariableSetting> q2)
        {
            var q3 = (from tb1 in q2
                join tb2 in _dbContext.BannerResx
                    on new {Name = tb1.ResxName, tb1.VarType} equals new {tb2.Name, tb2.VarType}
                    into g1
                from tb2 in g1.DefaultIfEmpty(new BannerResxEntity()
                {
                    Name = tb1.ResxName,
                    VarType = tb1.VarType,
                    Content = ""
                })
                select new VariableResxSetting
                {
                    VarName = tb1.VarName,
                    VarType = tb1.VarType,
                    ResxName = tb1.ResxName,
                    Content = tb2.Content,
                    IsoLangCode = tb2.IsoLangCode,
                }).ToList();
            return q3;
        }

        private static List<TemplateVariableSetting> GetTemplateVariablesSettings(TemplateBannerData banner)
        {
            var q2 = from tb1 in banner.BannerVariables
                join tb2 in banner.TemplateVariables on tb1.VarName equals tb2.Name
                    into g1
                from tb2 in g1.DefaultIfEmpty(new TemplateVariable()
                {
                    Name = tb1.VarName,
                    VarType = string.Empty
                })
                select new TemplateVariableSetting
                {
                    VarName = tb2.Name,
                    VarType = tb2.VarType,
                    ResxName = tb1.ResxName
                };
            return q2.ToList();
        }

        private List<TemplateBannerData> GetBannersData()
        {
            var q1 = from tb1 in _dbContext.Banners
                join tb2 in _dbContext.BannerTemplates on tb1.TemplateName equals tb2.TemplateName into g1
                from tb2 in g1.DefaultIfEmpty(new BannerTemplateEntity()
                {
                    TemplateName = tb1.TemplateName,
                    VariablesData = "{}"
                })
                select new TemplateBannerData
                {
                    Id = tb1.Id,
                    TemplateName = tb1.TemplateName,
                    Name = tb1.Name,
                    OrderId = tb1.OrderId,
                    TemplateVariables = BannerTemplate.ParseVariablesData(tb2.VariablesData),
                    BannerVariables = Banner.ParseVariableOptions(tb1.VariableOptions),
                };

            return q1.ToList();
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