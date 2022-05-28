using System.Collections.Immutable;
using System.Text.Json;
using PizzaWeb.Models.Helpers;
using T1.Standard.Common;

namespace PizzaWeb.Models.Banner;

public class BannerTemplate
{
    public int Id { get; set; }
    public string TemplateName { get; set; } = "";
    public string TemplateContent { get; set; } = "";
    public List<TemplateVariable> Variables { get; set; } = new List<TemplateVariable>();

    public static BannerTemplate From(BannerTemplateEntity entity)
    {
        var row = ValueHelper.CopyData(entity, new BannerTemplate());
        ParseVariablesJson(entity.VariablesJson);
        return row;
    }

    public static List<TemplateVariable> ParseVariablesJson(string? templateVariablesData)
    {
        if (string.IsNullOrEmpty(templateVariablesData))
        {
            return new List<TemplateVariable>();
        }

        var sp = ServiceLocator.Current;
        var jsonConvert = sp.GetService<IJsonConverter>();
        var dict = jsonConvert.Deserialize<Dictionary<string, string>>(templateVariablesData);

        return dict.Select(x => new TemplateVariable
        {
            Name = x.Key,
            VarType = x.Value
        }).ToList();
    }
}