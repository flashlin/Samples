using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryApp.Models.Clients;
using QueryApp.Models.Services;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalApiController : ControllerBase
{
    private readonly IReportRepo _reportRepo;
    private readonly ILocalQueryClient _localQueryClient;

    public LocalApiController(IReportRepo reportRepo, ILocalQueryClient localQueryClient)
    {
        _localQueryClient = localQueryClient;
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