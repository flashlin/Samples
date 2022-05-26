using System.Collections.Immutable;
using System.Text.Json;

namespace PizzaWeb.Models.Banner;

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
