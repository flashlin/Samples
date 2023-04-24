using Microsoft.AspNetCore.Mvc;

namespace QueryWeb.Controllers.api;

[ApiController]
[Route("api/[controller]/[action]")]
public class HealthController : ControllerBase
{
    public IActionResult ReadinessProbe()
    {
        return Ok();
    }

    public IActionResult LivenessProbe()
    {
        return Ok();
    }
}