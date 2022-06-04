using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public class UpdateTemplateData
{
    public int Id { get; set; }
    public string TemplateName { get; set; } = string.Empty;
    public string TemplateContent { get; set; } = string.Empty;
    public Dictionary<string, TemplateVariable> Variables { get; set; } = new Dictionary<string, TemplateVariable>();
}