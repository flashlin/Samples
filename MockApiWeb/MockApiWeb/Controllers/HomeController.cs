using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Routing;
using MockApiWeb.Models;

namespace MockApiWeb.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;

    public HomeController(ILogger<HomeController> logger)
    {
        _logger = logger;
    }

    public IActionResult Index()
    {
        return View();
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

[Route("api/[controller]/[action]")]
[ApiController]
public class MockWebApiController : ControllerBase
{
    [HttpPost, HttpGet]
    public JsonResult ProcessRequest([FromBody] dynamic request)
    {
        return new JsonResult(new
        {
            Id = 123,
            Name = "Flash"
        });
    }
}

public class DynamicApiTransformer : DynamicRouteValueTransformer
{
    public DynamicApiTransformer()
    {
    }
 
    public override async ValueTask<RouteValueDictionary> TransformAsync(HttpContext httpContext, RouteValueDictionary values)
    {
        if (!values.ContainsKey("controller") || !values.ContainsKey("action"))
        {
            return values;
        }
 
        var controller = (string?)values["controller"];
        if (controller == null) return values;
        values["controller"] = "MockWebApi";  //controller;
 
        var action = (string?)values["action"];
        if (action == null) return values;
        values["action"] = "ProcessRequest";//action;
 
        return values;
    }
}