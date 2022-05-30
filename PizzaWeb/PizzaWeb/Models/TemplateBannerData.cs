using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models;

public class TemplateBannerData
{
    public int Id { get; set; }
    public string TemplateName { get; set; }
    public string Name { get; set; }
    public int OrderId { get; set; }
    public string TemplateVariablesJson { get; set; }
    public string BannerVariablesJson { get; set; }
}