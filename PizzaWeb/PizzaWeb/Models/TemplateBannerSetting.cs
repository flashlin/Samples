using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models;

public class TemplateBannerSetting
{
    public int Id { get; set; }
    public string TemplateName { get; set; } = String.Empty;
    public string BannerName { get; set; } = String.Empty;
    public int OrderId { get; set; }
    public List<TemplateVariable> TemplateVariables { get; set; } = new List<TemplateVariable>();
    public List<VariableOption> BannerVariables { get; set; } = new List<VariableOption>();
}