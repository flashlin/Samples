using System.ComponentModel.DataAnnotations.Schema;

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