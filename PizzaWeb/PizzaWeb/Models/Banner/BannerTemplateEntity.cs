using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models.Banner;

[Table("BannerTemplate")]
public class BannerTemplateEntity
{
	[Key]
	[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }
    public string TemplateName { get; set; } = string.Empty;
    public string TemplateContent { get; set; } = string.Empty;
    //public string VariablesJson { get; set; } = "{}";
    public List<TemplateVariable> VariablesJson { get; set; } = new List<TemplateVariable>();
    public DateTime LastModifiedTime { get; set; }
}