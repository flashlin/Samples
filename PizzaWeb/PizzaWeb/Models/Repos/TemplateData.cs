using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models.Repos;

public class TemplateData
{
    public int Id { get; set; }
    public string TemplateName { get; set; } = string.Empty;
    public string TemplateContent { get; set; } = string.Empty;
    public List<TemplateVariable> Variables { get; set; } = new List<TemplateVariable>();
}