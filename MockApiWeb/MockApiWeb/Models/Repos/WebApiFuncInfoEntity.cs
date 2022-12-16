using System.ComponentModel.DataAnnotations.Schema;
using MockApiWeb.Controllers;

namespace MockApiWeb.Models.Repos;

[Table("WebApiFuncInfos")]
public class WebApiFuncInfoEntity
{
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }
    public string ProductName { get; set; } = string.Empty;
    public string ControllerName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public WebApiAccessMethodType Method { get; set; }
    public string ResponseContent { get; set; } = string.Empty;
    public int ResponseStatus { get; set; }
}