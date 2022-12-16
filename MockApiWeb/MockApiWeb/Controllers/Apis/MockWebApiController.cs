using Microsoft.AspNetCore.Mvc;
using MockApiWeb.Models.Repos;
using MockApiWeb.Models.Requests;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MockWebApiController : ControllerBase
{
    private IMockDbRepo _mockDbRepo;

    public MockWebApiController(IMockDbRepo mockDbRepo)
    {
        _mockDbRepo = mockDbRepo;
    }
    
    
    [HttpPost, HttpGet]
    public ContentResult ProcessRequest(MockWebApiRequest req)
    {
        var responseSettings = _mockDbRepo.GetWebApiResponseSetting(req);

        return new ContentResult
        {
            Content = responseSettings.ResponseContent,
            ContentType = "application/json"
        };
    }
}