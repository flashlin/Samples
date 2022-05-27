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
        if (entity.VariablesData != null)
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            row.Variables = jsonConvert.Deserialize<List<TemplateVariable>>(entity.VariablesData) ??
                            new List<TemplateVariable>();
        }
        return row;
    }
}