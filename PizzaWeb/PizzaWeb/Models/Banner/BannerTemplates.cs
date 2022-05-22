using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;

namespace PizzaWeb.Models.Banner;

[Table("BannerTemplates")]
public class BannerTemplateEntity
{
	public int Id { get; set; }
	public string TemplateName { get; set; }
	public string TemplateContent { get; set; }
	public string? VariablesData { get; set; }
	public DateTime LastModifiedTime { get; set; }
}

public class BannerTemplate
{
	public int Id { get; set; }
	public string TemplateName { get; set; } = "";
	public string TemplateContent { get; set; } = "";
	public Dictionary<string, TemplateVariable> Variables { get; set; } = new Dictionary<string, TemplateVariable>();

	public Dictionary<string, TemplateVariable> GetVariables(string variablesData)
	{
		var jsonOptions = new JsonSerializerOptions
		{
			AllowTrailingCommas = true,
			PropertyNamingPolicy = JsonNamingPolicy.CamelCase
		};
		var variablesList = JsonSerializer.Deserialize<List<TemplateVariable>>(variablesData, jsonOptions);
		if (variablesList == null)
		{
			return new Dictionary<string, TemplateVariable>();
		}
		return variablesList.ToDictionary(x => x.Name);
	}

	public string GetVariablesData()
	{
		var jsonOptions = new JsonSerializerOptions
		{
			AllowTrailingCommas = true,
			PropertyNamingPolicy = JsonNamingPolicy.CamelCase
		};
		return JsonSerializer.Serialize(Variables.Values, jsonOptions);
	}
}

public class TemplateVariable
{
	public string Name { get; set; } = "";
	public TemplateVariableType VarType { get; set; }
}

public enum TemplateVariableType
{
	String,
	Number
}