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
    private readonly IQueryClient _queryClient;

    public LocalApiController(IReportRepo reportRepo, IQueryClient queryClient)
    {
        _queryClient = queryClient;
        _reportRepo = reportRepo;
    }
    
    [HttpPost]
    public OkResult Knock(KnockRequest req)
    {
        _queryClient.KnockAsync(req);
        return Ok();
    }

    [HttpPost]
    public List<string> GetAllTableNames()
    {
        return _reportRepo.GetAllTableNames();
    }
}