using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace PizzaWeb.Models.Banner;

[Table("VariableShelf")]
public class VariableShelfEntity
{
	[Key]
	[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
	public int Id { get; set; } = 0;
    public Guid Uid { get; set; }
    public string VarName { get; set; } = string.Empty;
    public string ResxName { get; set; } = string.Empty;
    public string IsoLangCode { get; set; } = "en-US";
    public string Content { get; set; } = String.Empty;
}