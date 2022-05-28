using Microsoft.EntityFrameworkCore;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Controllers;

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

    public List<TemplateBannerData> GetBannersData()
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

    public List<TemplateVariableSetting> GetTemplateVariablesSettings(TemplateBannerData banner)
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

    public List<VariableResxSetting> GetAllVariableSettings(List<TemplateVariableSetting> q2)
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

    public List<BannerVariable> GetBannerVariables(List<VariableResxSetting> q3)
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

    public IEnumerable<Banner> GetAllBanners()
    {
        var banners = this.GetBannersData();
        foreach (var banner in banners)
        {
            var q2 = this.GetTemplateVariablesSettings(banner);
            var q3 = this.GetAllVariableSettings(q2);
            var q4 = this.GetBannerVariables(q3);
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
}