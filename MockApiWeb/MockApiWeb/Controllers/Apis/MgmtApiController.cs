using AutoMapper;
using Microsoft.AspNetCore.Mvc;
using MockApiWeb.Models.DataConstraints;
using MockApiWeb.Models.Parameters;
using MockApiWeb.Models.Repos;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MgmtApiController : ControllerBase
{
    private readonly IMockDbRepo _mockDbRepo;
    private readonly IMapper _mapper;

    public MgmtApiController(IMockDbRepo mockDbRepo, IMapper mapper)
    {
        _mapper = mapper;
        _mockDbRepo = mockDbRepo;
    }
    
    [HttpPost]
    public ActionResult CreateDefaultResponse(MockWebApiSimpleSettingRequest req)
    {
        _mockDbRepo.AddMockWebApiSimpleSetting(_mapper.Map<MockWebApiSimpleSettingParameters>(req));
        return Ok();
    }

    [HttpPost]
    public DefaultResponsePageData QueryDefaultResponsePage(GetWebApiSimpleSettingRequest req)
    {
        return _mockDbRepo.QueryDefaultResponsePage(req);
    }
}