using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;

namespace PizzaWeb.Models.Banner;

[Table("BannerTemplates")]
public class BannerTemplateEntity
{
    public string Id { get; set; }
    public string TemplateContent { get; set; }
    public string? VariablesData { get; set; }

    public Dictionary<string, TemplateVariable> GetVariables()
    {
        var jsonOptions = new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        var variablesList = JsonSerializer.Deserialize<List<TemplateVariable>>(VariablesData, jsonOptions);
        if (variablesList == null)
        {
            return new Dictionary<string, TemplateVariable>();
        }

        return variablesList.ToDictionary(x => x.Name);
    }
}

public class TemplateVariable
{
    public string Name { get; set; }
    public TempVariableType TempVarType { get; set; }
}

public enum TempVariableType
{
    String,
    Number
}