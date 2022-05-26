using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models.Banner;

[Table("BannerVariables")]
public class BannerVariableEntity
{
    public int Id { get; set; }
    public int TemplateId { get; set; }
    public string VariableName { get; set; }
    public int ResxId { get; set; }
}