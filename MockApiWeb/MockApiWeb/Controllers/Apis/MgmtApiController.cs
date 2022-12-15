using Microsoft.AspNetCore.Mvc;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MgmtApiController : ControllerBase
{
    [HttpPost]
    public JsonResult CreateDefaultResponse(string swaggerUrl)
    {
        return new JsonResult(new
        {
            Id = 133,
            Name = "Flashcc"
        });
    }
}