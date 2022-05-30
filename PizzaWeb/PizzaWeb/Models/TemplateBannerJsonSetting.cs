using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models;

public class TemplateBannerJsonSetting
{
    public int Id { get; set; }
    public string TemplateName { get; set; } = String.Empty;
    public string BannerName { get; set; } = String.Empty;
    public int OrderId { get; set; }
    public string TemplateVariablesJson { get; set; } = "{}";
    public string BannerVariablesJson { get; set; } = "{}";
}