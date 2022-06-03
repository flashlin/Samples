using System.Data;
using Dapper;
using Microsoft.EntityFrameworkCore;
using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using T1.Standard.Common;
using T1.Standard.Data;
using T1.Standard.Linq;
using BannerVariable = PizzaWeb.Models.Banner.BannerVariable;

namespace PizzaWeb.Models.Repos;

public class PizzaRepo : IPizzaRepo
{
    private readonly PizzaDbContext _dbContext;
    private readonly IJsonConverter _jsonConverter;

    public PizzaRepo(PizzaDbContext dbContext, IJsonConverter jsonConverter)
    {
        _jsonConverter = jsonConverter;
        _dbContext = dbContext;
    }

    public List<BannerTemplate> GetAllBannerTemplates()
    {
        var q1 = from tb1 in _dbContext.BannerTemplates.AsNoTracking()
            select BannerTemplate.From(tb1);
        return q1.ToList();
    }

    public List<BannerTemplateEntity> GetTemplateContents(string[] templateNames)
    {
        return _dbContext.BannerTemplates
            .Where(x => templateNames.Contains(x.TemplateName))
            .Select(x => new BannerTemplateEntity
            {
                Id = x.Id,
                TemplateName = x.TemplateName,
                TemplateContent = x.TemplateContent,
                Variables = x.Variables
            })
            .ToList();
    }

    public IQueryable<TemplateBannerSetting> QueryBannerSettingData()
    {
        return from tb1 in _dbContext.Banners.AsNoTracking()
            let tb2 = _dbContext.BannerTemplates
                .AsNoTracking()
                .FirstOrDefault(x => x.TemplateName == tb1.TemplateName)
            select new TemplateBannerSetting
            {
                Id = tb1.Id,
                TemplateName = tb1.TemplateName,
                BannerName = tb1.BannerName,
                OrderId = tb1.OrderId,
                TemplateVariables = tb2.Variables,
                BannerVariables = tb1.VariableOptions,
            };
    }

    private IEnumerable<TemplateVariableSetting> QueryTemplateVariablesSettings(TemplateBannerSetting banner)
    {
        return from tb1 in banner.BannerVariables
            join tb2 in banner.TemplateVariables
                on tb1.VarName equals tb2.VarName
                into g1
            from tb2 in g1.DefaultIfEmpty(new TemplateVariable()
            {
                VarName = tb1.VarName,
                VarType = string.Empty
            })
            select new TemplateVariableSetting
            {
                VarName = tb2.VarName,
                VarType = tb2.VarType,
                ResxName = tb1.ResxName
            };
    }

    private IEnumerable<VariableResxSetting> QueryAllVariableSettings(
        IEnumerable<TemplateVariableSetting> templateVariableSettings)
    {
        return from tb1 in templateVariableSettings
            join tb2 in _dbContext.BannerResx
                on new {Name = tb1.ResxName, tb1.VarType} equals new {Name = tb2.ResxName, tb2.VarType}
                into g1
            from tb2 in g1.DefaultIfEmpty(new BannerResxEntity()
            {
                ResxName = tb1.ResxName,
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

    public List<BannerSetting> GetBannersSetting(GetBannersSettingReq req)
    {
        return QueryAllBanners()
            .Where(x => x.TemplateName == req.TemplateName)
            .ToList();
    }

    private IEnumerable<BannerSetting> QueryAllBanners()
    {
        var banners = this.QueryBannerSettingData().ToList();
        return from banner in banners
            let variables = QueryTemplateVariableOptions(banner)
            select new BannerSetting
            {
                Id = banner.Id,
                Name = banner.BannerName,
                TemplateName = banner.TemplateName,
                OrderId = banner.OrderId,
                Variables = variables.ToList()
            };
    }

    private IEnumerable<BannerVariable> QueryTemplateVariableOptions(TemplateBannerSetting banner)
    {
        var templateVariablesSettings = this.QueryTemplateVariablesSettings(banner);
        var variableSettings = this.QueryAllVariableSettings(templateVariablesSettings);
        var variables = this.QueryBannerVariables(variableSettings);
        return variables;
    }

    public void AddBannerTemplate(AddBannerTemplateReq req)
    {
        _dbContext.BannerTemplates.Add(new BannerTemplateEntity()
        {
            TemplateName = req.TemplateName,
            TemplateContent = req.TemplateContent,
            Variables = req.Variables.Values.ToList(),
            LastModifiedTime = DateTime.UtcNow
        });
        _dbContext.SaveChanges();
    }

    public void AddBanner(AddBannerReq req)
    {
        _dbContext.Banners.Add(new BannerEntity()
        {
            BannerName = req.BannerName,
            TemplateName = req.TemplateName,
            OrderId = req.OrderId,
            VariableOptions = req.VariablesOptions.Values.ToList(),
            LastModifiedTime = DateTime.UtcNow
        });
        _dbContext.SaveChanges();
    }

    public void UpdateBannerTemplate(BannerTemplate req)
    {
        var bannerTemplate = _dbContext.BannerTemplates.Find(req.Id)!;
        bannerTemplate.LastModifiedTime = DateTime.Now;
        bannerTemplate.TemplateContent = req.TemplateContent;
        bannerTemplate.Variables = req.Variables;
        _dbContext.BannerTemplates.Update(bannerTemplate);
        _dbContext.SaveChanges();
    }

    public void ApplyBanner(string bannerName)
    {
        _dbContext.Database.ExecuteSqlRaw("delete BannerShelf");
        _dbContext.Database.ExecuteSqlRaw("delete VariableShelf");

        var bannerSettings = (
            from tb1 in _dbContext.Banners.AsNoTracking()
            join tb2 in _dbContext.BannerTemplates.AsNoTracking() on tb1.TemplateName equals tb2.TemplateName
            where tb1.BannerName == bannerName
            select new
            {
                TemplateName = tb2.TemplateName,
                TemplateContent = tb2.TemplateContent,
                TemplateVariables = tb2.Variables,
                BannerName = tb1.BannerName,
                OrderId = tb1.OrderId,
                BannerVariableOptions = tb1.VariableOptions,
                Uid = Guid.NewGuid(),
            }).ToArray();

        var templateVariables = (
            from tb1 in bannerSettings
            from tb2 in tb1.TemplateVariables
            select tb2).ToArray();

        var variableOptions = (from tb1 in bannerSettings
            from tb2 in tb1.BannerVariableOptions
            select tb2).ToArray();

        var resxNames = (
            from tb1 in templateVariables
            join tb2 in variableOptions on tb1.VarName equals tb2.VarName
            select new
            {
                VarName = tb1.VarName,
                ResxName = tb2.ResxName,
                VarType = tb1.VarType,
            }).ToArray();

        var resxNamesDict = (
            from tb1 in resxNames
            select new Dictionary<string, object>
            {
                {"ResxName", tb1.ResxName},
                {"VarType", tb1.VarType},
            }).ToArray();

        var resxNamesTable = resxNamesDict.ToDataTable();
        var conn = _dbContext.Database.GetDbConnection();
        var bannerResx = conn.Query<BannerResxEntity>(
            @"SP_GetResxNames", new
            {
                ResxNames = resxNamesTable
            }, commandType: CommandType.StoredProcedure).ToArray();

        var bannerShelf = new List<BannerShelfEntity>();
        var varShelf = new List<VariableShelfEntity>();

        foreach (var setting in bannerSettings)
        {
            bannerShelf.Add(new BannerShelfEntity
            {
                Uid = setting.Uid,
                BannerName = setting.BannerName,
                TemplateName = setting.TemplateName,
                TemplateContent = setting.TemplateContent,
                OrderId = setting.OrderId,
            });

            var bannerVars = (
                from tb1 in setting.TemplateVariables
                join tb2 in setting.BannerVariableOptions on tb1.VarName equals tb2.VarName
                select new
                {
                    VarName = tb1.VarName,
                    ResxName = tb2.ResxName,
                    VarType = tb1.VarType,
                }).ToArray();

            var varSettings = (
                from tb1 in bannerResx
                join tb2 in bannerVars on new {tb1.ResxName, tb1.VarType} equals new {tb2.ResxName, tb2.VarType}
                select new VariableShelfEntity()
                {
                    Uid = setting.Uid,
                    VarName = tb2.VarName,
                    ResxName = tb1.ResxName,
                    IsoLangCode = tb1.IsoLangCode,
                    Content = tb1.Content
                }).ToArray();

            varShelf.AddRange(varSettings);
        }

        _dbContext.VariableShelf.AddRange(varShelf);
        _dbContext.SaveChanges();

        _dbContext.BannerShelf.AddRange(bannerShelf);
        _dbContext.SaveChanges();
    }

    public List<BannerTemplateData> GetBannersData(GetBannersDataReq req)
    {
        var banners = (
            from tb1 in _dbContext.BannerShelf.AsNoTracking()
            join tb2 in _dbContext.VariableShelf.AsNoTracking() on tb1.Uid equals tb2.Uid
            where tb1.BannerName == req.BannerName && tb2.IsoLangCode == req.IsoLangCode
            select new
            {
                Uid = tb1.Uid,
                BannerName = req.BannerName,
                TemplateName = tb1.TemplateName,
                TemplateContent = tb1.TemplateContent,
                OrderId = tb1.OrderId,
                IsoLangCode = tb2.IsoLangCode,
                VarName = tb2.VarName,
                ResxName = tb2.ResxName,
                ResxContent = tb2.Content,
            }).ToArray();

        var bannersData = from tb1 in banners
            group tb1 by tb1.Uid
            into g1
            let tb2 = g1.First()
            select new BannerTemplateData
            {
                Uid = tb2.Uid,
                BannerName = tb2.BannerName,
                TemplateName = tb2.TemplateName,
                TemplateContent = tb2.TemplateContent,
                Variables = g1
                    .OrderBy(x => x.OrderId)
                    .Select(x => new BannerData()
                    {
                        VarName = x.VarName,
                        ResxName = x.ResxName,
                        Content = x.ResxContent
                    })
                    .ToList()
            };

        return bannersData.ToList();
    }
}

public class AddBannerTemplateReq
{
    public string TemplateName { get; set; } = string.Empty;
    public string TemplateContent { get; set; } = string.Empty;
    public Dictionary<string, TemplateVariable> Variables { get; set; } = new Dictionary<string, TemplateVariable>();
}