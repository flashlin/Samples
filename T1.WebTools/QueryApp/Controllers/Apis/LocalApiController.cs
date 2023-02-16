using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryApp.Models.Services;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalApiController : ControllerBase
{
    private readonly IReportRepo _reportRepo;

    public LocalApiController(IReportRepo reportRepo)
    {
        _reportRepo = reportRepo;
    }
    
    [HttpPost]
    public OkResult Knock(KnockRequest req)
    {
        return Ok();
    }

    [HttpPost]
    public List<string> GetAllTableNames()
    {
        return _reportRepo.GetAllTableNames();
    }
}