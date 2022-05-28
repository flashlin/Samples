using Microsoft.EntityFrameworkCore;
using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public class PizzaRepo
{
    private readonly PizzaDbContext _dbContext;

    public PizzaRepo(PizzaDbContext dbContext)
    {
        _dbContext = dbContext;
    }

    public List<BannerTemplate> GetAllBannerTemplates()
    {
        var bannerTemplates = _dbContext.BannerTemplates
            .AsNoTracking()
            .ToList();
        return bannerTemplates.Select(x => BannerTemplate.From(x)).ToList();
    }

    private IQueryable<TemplateBannerData> QueryBannersData()
    {
        return from tb1 in _dbContext.Banners
            join tb2 in _dbContext.BannerTemplates on tb1.TemplateName equals tb2.TemplateName into g1
            from tb2 in g1.DefaultIfEmpty(new BannerTemplateEntity()
            {
                TemplateName = tb1.TemplateName,
                VariablesJson = "{}"
            })
            select new TemplateBannerData
            {
                Id = tb1.Id,
                TemplateName = tb1.TemplateName,
                Name = tb1.Name,
                OrderId = tb1.OrderId,
                TemplateVariables = BannerTemplate.ParseVariablesJson(tb2.VariablesJson),
                BannerVariables = Banner.Banner.ParseVariableOptionsJson(tb1.VariableOptions),
            };
    }

    private IEnumerable<TemplateVariableSetting> QueryTemplateVariablesSettings(TemplateBannerData banner)
    {
        return from tb1 in banner.BannerVariables
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
    }

    private IEnumerable<VariableResxSetting> QueryAllVariableSettings(
        IEnumerable<TemplateVariableSetting> templateVariableSettings)
    {
        return from tb1 in templateVariableSettings
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
            };
    }

    private IEnumerable<BannerVariable> QueryBannerVariables(IEnumerable<VariableResxSetting> variablesSettings)
    {
        return from tb1 in variablesSettings
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
            };
    }

    public List<Banner.Banner> GetAllBanners()
    {
        return QueryAllBanners().ToList();
    }

    private IEnumerable<Banner.Banner> QueryAllBanners()
    {
        var banners = this.QueryBannersData().ToList();
        foreach (var banner in banners)
        {
            var templateVariablesSettings = this.QueryTemplateVariablesSettings(banner);
            var variableSettings = this.QueryAllVariableSettings(templateVariablesSettings);
            var variables = this.QueryBannerVariables(variableSettings);
            yield return new Banner.Banner
            {
                Id = banner.Id,
                Name = banner.Name,
                TemplateName = banner.TemplateName,
                OrderId = banner.OrderId,
                Variables = variables.ToList()
            };
        }
    }
}