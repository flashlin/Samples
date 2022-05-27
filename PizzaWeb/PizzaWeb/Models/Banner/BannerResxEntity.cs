using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models.Banner;

[Table("Resx")]
public class BannerResxEntity
{
	[Key]
	[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
	public int Id { get; set; } = 0;

    public string IsoLangCode { get; set; } = "en-US";
    public string VarType { get; set; } = "String";
    public string Name { get; set; } = "";
    public string Content { get; set; } = "";
}