using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models.Banner;

[Table("BannerShelf")]
public class BannerShelfEntity
{
    [Key]
    public Guid Uid { get; set; }
    public string TemplateName { get; set; } = String.Empty;
    public string TemplateContent { get; set; } = String.Empty;
    public int OrderId { get; set; }
    public string BannerName { get; set; } = String.Empty;
}