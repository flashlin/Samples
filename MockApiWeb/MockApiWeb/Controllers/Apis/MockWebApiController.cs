using Microsoft.AspNetCore.Mvc;
using MockApiWeb.Models.Requests;

namespace MockApiWeb.Controllers.Apis;

[Route("api/[controller]/[action]")]
[ApiController]
public class MockWebApiController : ControllerBase
{
    public MockWebApiController()
    {
        
    }
    
    
    [HttpPost, HttpGet]
    public JsonResult ProcessRequest(MockWebApiRequest req)
    {
        return new JsonResult(new
        {
            Id = 123,
            Name = "Flash",
            Product = req.ProductName,
            RequestBody = req.RequestBody,
            RequestQueryString = req.RequestQueryString
        });
    }
}