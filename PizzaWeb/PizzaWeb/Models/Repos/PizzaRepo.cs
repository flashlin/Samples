using Microsoft.EntityFrameworkCore;
using PizzaWeb.Controllers;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;

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
}

public class PizzaRepo : IPizzaRepo
{
    private readonly PizzaDbContext _dbContext;
    private IJsonConverter _jsonConverter;

    public PizzaRepo(PizzaDbContext dbContext, IJsonConverter jsonConverter)
    {
        _jsonConverter = jsonConverter;
        _dbContext = dbContext;
    }

    public List<BannerTemplate> GetAllBannerTemplates()
    {
        var bannerTemplates = _dbContext.BannerTemplates
            .AsNoTracking()
            .ToList();
        return bannerTemplates.Select(x => BannerTemplate.From(x)).ToList();
    }

    public IQueryable<TemplateBannerJsonSetting> QueryBannerJsonSettingData()
    {
        return from tb1 in _dbContext.Banners.AsNoTracking()
            let tb2 = _dbContext.BannerTemplates
                .AsNoTracking()
                .FirstOrDefault(x => x.TemplateName == tb1.TemplateName)
            select new TemplateBannerJsonSetting
            {
                Id = tb1.Id,
                TemplateName = tb1.TemplateName,
                BannerName = tb1.BannerName,
                OrderId = tb1.OrderId,
                TemplateVariablesJson = tb2.VariablesJson ?? "{}",
                BannerVariablesJson = tb1.VariableOptionsJson ?? "{}",
            };
    }

    private IEnumerable<TemplateVariableSetting> QueryTemplateVariablesSettings(TemplateBannerJsonSetting bannerJson)
    {
        return from tb1 in bannerJson.BannerVariablesJson.ToVariableOptionsList()
            join tb2 in bannerJson.TemplateVariablesJson.ToTemplateVariablesList()
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
        var banners = this.QueryBannerJsonSettingData().ToList();
        foreach (var bannerSetting in QueryBannerSettings(banners)) yield return bannerSetting;
    }

    public IEnumerable<BannerSetting> QueryBannerSettings(List<TemplateBannerJsonSetting> banners)
    {
        foreach (var banner in banners)
        {
            var templateVariablesSettings = this.QueryTemplateVariablesSettings(banner);
            var variableSettings = this.QueryAllVariableSettings(templateVariablesSettings);
            var variables = this.QueryBannerVariables(variableSettings);
            yield return new BannerSetting
            {
                Id = banner.Id,
                Name = banner.BannerName,
                TemplateName = banner.TemplateName,
                OrderId = banner.OrderId,
                Variables = variables.ToList()
            };
        }
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
                VariablesJson = x.VariablesJson
            })
            .ToList();
    }

    public void AddBannerTemplate(AddBannerTemplateReq req)
    {
        _dbContext.BannerTemplates.Add(new BannerTemplateEntity()
        {
            TemplateName = req.TemplateName,
            TemplateContent = req.TemplateContent,
            VariablesJson = req.Variables.ToJson(),
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
            VariableOptionsJson = req.VariablesOptions.ToJson(),
            LastModifiedTime = DateTime.UtcNow
        });
        _dbContext.SaveChanges();
    }

    public void UpdateBannerTemplate(BannerTemplate req)
    {
        var bannerTemplate = _dbContext.BannerTemplates.Find(req.Id)!;
        bannerTemplate.LastModifiedTime = DateTime.Now;
        bannerTemplate.TemplateContent = req.TemplateContent;
        bannerTemplate.VariablesJson = _jsonConverter.Serialize(req.Variables);
        _dbContext.BannerTemplates.Update(bannerTemplate);
        _dbContext.SaveChanges();
    }

    public void ApplyBanner(string bannerName)
    {
        _dbContext.Database.ExecuteSqlRaw("delete BannerShelf");
        _dbContext.Database.ExecuteSqlRaw("delete VariableShelf");
        
        var settings = (from tb1 in _dbContext.Banners.AsNoTracking()
            join tb2 in _dbContext.BannerTemplates.AsNoTracking() on tb1.TemplateName equals tb2.TemplateName
            where tb1.BannerName == bannerName
            select new 
            {
                TemplateName = tb2.TemplateName,
                TemplateContent = tb2.TemplateContent,
                TemplateVariables = tb2.VariablesJson.ToTemplateVariablesList(),
                BannerName = tb1.BannerName,
                OrderId = tb1.OrderId,
                BannerVariableOptions = tb1.VariableOptionsJson.ToVariableOptionsList(),
                Uid = Guid.NewGuid(),
            }).ToList();

        var resxNames = (from tb1 in settings
            from tb2 in tb1.BannerVariableOptions
            group tb2 by tb2.ResxName into g1
            select g1.Key).ToList();
        
        var resx = (from tb1 in _dbContext.BannerResx.AsNoTracking()
            where resxNames.Contains(tb1.ResxName)
            select new BannerResxEntity
            {
                ResxName = tb1.ResxName,
                IsoLangCode = tb1.IsoLangCode,
                Content = tb1.Content,
            }).ToList();

        var bannerShelf = new List<BannerShelfEntity>();
        var varShelf = new List<VariableShelfEntity>();

        foreach (var setting in settings)
        {
            bannerShelf.Add(new BannerShelfEntity
            {
                Uid = setting.Uid,
                BannerName = setting.BannerName,
                TemplateName = setting.TemplateName,
                TemplateContent = setting.TemplateContent,
                OrderId = setting.OrderId,
            });

            var varSettings = from tb1 in setting.TemplateVariables
                join tb2 in setting.BannerVariableOptions on tb1.VarName equals tb2.VarName
                let tb3 = (from tb3 in resx
                    where tb3.ResxName == tb2.ResxName && tb3.VarType == tb1.VarType
                    select tb3).ToArray()
                from tb4 in tb3
                select new VariableShelfEntity()
                {
                    Uid = setting.Uid,
                    VarName = tb1.VarName,
                    ResxName = tb2.ResxName,
                    IsoLangCode = tb4.IsoLangCode,
                    Content = tb4.Content
                };
            
            varShelf.AddRange(varSettings.ToArray());
        }
        
        _dbContext.VariableShelf.AddRange(varShelf);
        _dbContext.SaveChanges();
            
        _dbContext.BannerShelf.AddRange(bannerShelf);
        _dbContext.SaveChanges();
    }
}

public class AddBannerTemplateReq
{
    public string TemplateName { get; set; } = string.Empty;
    public string TemplateContent { get; set; } = string.Empty;
    public Dictionary<string, TemplateVariable> Variables { get; set; } = new Dictionary<string, TemplateVariable>();
}