using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
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
    public string ResponseContent { get; set; } = string.Empty;
    public int ResponseStatus { get; set; }

    public ObjectResult GetResponseResult()
    {
        if (!string.IsNullOrEmpty(ResponseContent))
        {
            var responseObject = JsonSerializer.Deserialize<object>(ResponseContent);
            return new ObjectResult(responseObject)
            {
                StatusCode = ResponseStatus
            };
        }
        return new ObjectResult(null)
        {
            StatusCode = ResponseStatus
        };
    }
}