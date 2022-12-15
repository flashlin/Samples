using Microsoft.AspNetCore.Mvc;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MockWebApiController : ControllerBase
{
    [HttpPost, HttpGet]
    public JsonResult ProcessRequest([FromBody] dynamic? request)
    {
        return new JsonResult(new
        {
            Id = 123,
            Name = "Flash"
        });
    }
}

[Route("api/[controller]/[action]")]
[ApiController]
public class MgmtApiController : ControllerBase
{
    public JsonResult ProcessRequest([FromBody] dynamic? request)
    {
        return new JsonResult(new
        {
            Id = 133,
            Name = "Flashcc"
        });
    }
}
