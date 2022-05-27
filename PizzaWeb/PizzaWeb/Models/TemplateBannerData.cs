using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models;

public class TemplateBannerData
{
    public int Id { get; set; }
    public string TemplateName { get; set; }
    public string Name { get; set; }
    public int OrderId { get; set; }
    public List<TemplateVariable> TemplateVariables { get; set; }
    public List<TemplateVariableValue> BannerVariables { get; set; }
}