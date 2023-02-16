using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalApiController : ControllerBase
{
    [HttpPost]
    public OkResult Knock(KnockRequest req)
    {
        return Ok();
    }
}