using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;
using AutoMapper;
using Microsoft.AspNetCore.Mvc;
using MockApiWeb.Models.DataConstraints;
using MockApiWeb.Models.Parameters;
using MockApiWeb.Models.Repos;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MockWebApiController : ControllerBase
{
    private readonly IMockDbRepo _mockDbRepo;
    private readonly IMapper _mapper;

    public MockWebApiController(IMockDbRepo mockDbRepo, IMapper mapper)
    {
        _mapper = mapper;
        _mockDbRepo = mockDbRepo;
    }
    
    [HttpPost, HttpGet]
    public ActionResult ProcessRequest(MockWebApiRequest req)
    {
        var responseSettings = _mockDbRepo.GetWebApiResponseSetting(_mapper.Map<MockWebApiParameters>(req));
        return responseSettings.GetResponseResult();

        if (responseSettings.ResponseStatus != 200)
        {
            return responseSettings.GetResponseResult();
        }
        
        return new ContentResult
        {
            Content = responseSettings.ResponseContent,
            ContentType = "application/json"
        };
    }
}