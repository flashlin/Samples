using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryApp.Models.Services;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalQueryApiController : ControllerBase
{
    private readonly IReportRepo _reportRepo;
    private readonly ILocalEnvironment _localEnvironment;

    public LocalQueryApiController(IReportRepo reportRepo, ILocalEnvironment localEnvironment)
    {
        _localEnvironment = localEnvironment;
        _reportRepo = reportRepo;
    }
    
    [HttpPost]
    public KnockResponse Knock(KnockRequest req)
    {
        return new KnockResponse
        {
            AppUid = _localEnvironment.AppUid,
        };
    }

    [HttpPost]
    public List<string> GetAllTableNames()
    {
        return _reportRepo.GetAllTableNames();
    }
}