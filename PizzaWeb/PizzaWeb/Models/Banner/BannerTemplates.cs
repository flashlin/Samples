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
        var row = ValueHelper.CopyData(entity, new BannerTemplate(), 1);
        //row.Variables = entity.Variables;
        return row;
    }
}

public static class ParseJsonExtension
{
    public static List<TemplateVariable> ToTemplateVariablesList(this string? templateVariablesJson)
    {
        if (string.IsNullOrEmpty(templateVariablesJson))
        {
            return new List<TemplateVariable>();
        }

        var sp = ServiceLocator.Current;
        var jsonConvert = sp.GetService<IJsonConverter>();
        var dict = jsonConvert.Deserialize<Dictionary<string, TemplateVariable>>(templateVariablesJson);

        return dict.Values.ToList();
    }
    
    public static List<VariableOption> ToVariableOptionsList(this string variableOptions)
    {
        var dict = new Dictionary<string, VariableOption>();
        if (!string.IsNullOrEmpty(variableOptions))
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            dict = jsonConvert.Deserialize<Dictionary<string, VariableOption>>(variableOptions);
        }
        return dict.Values.ToList();
    }

    public static string ToJson(this Dictionary<string,TemplateVariable>? obj)
    {
        if (obj == null)
        {
            return "{}";
        }
        var sp = ServiceLocator.Current;
        var jsonConvert = sp.GetService<IJsonConverter>();
        return jsonConvert.Serialize(obj);
    }

    public static string ToJson(this Dictionary<string, VariableOption> dict)
    {
        var sp = ServiceLocator.Current;
        var jsonConvert = sp.GetService<IJsonConverter>();
        return jsonConvert.Serialize(dict);
    }
}