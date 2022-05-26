using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models.Banner;

[Table("BannerResx")]
public class BannerResxEntity
{
    public int Id { get; set; }
    public string Lang { get; set; }
    public string Name { get; set; }
    public string Content { get; set; }
    public DateTime LastModifiedTime { get; set; }
}