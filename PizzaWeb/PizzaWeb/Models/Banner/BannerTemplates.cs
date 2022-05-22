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
	public List<TemplateVariable> Variables { get; set; } = new List<TemplateVariable>();

	public List<TemplateVariable> GetVariables(string variablesData)
	{
		var jsonOptions = new JsonSerializerOptions
		{
			AllowTrailingCommas = true,
			PropertyNamingPolicy = JsonNamingPolicy.CamelCase
		};
		var variablesList = JsonSerializer.Deserialize<List<TemplateVariable>>(variablesData, jsonOptions);
		if (variablesList == null)
		{
			return new List<TemplateVariable>();
		}
		return variablesList;
	}

	public string GetVariablesData()
	{
		var jsonOptions = new JsonSerializerOptions
		{
			AllowTrailingCommas = true,
			PropertyNamingPolicy = JsonNamingPolicy.CamelCase
		};
		return JsonSerializer.Serialize(Variables, jsonOptions);
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