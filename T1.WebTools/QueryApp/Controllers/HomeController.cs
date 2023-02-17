using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;

namespace QueryApp.Controllers;

public class HomeController : Controller
{
    private readonly ILocalEnvironment _localEnvironment;

    public HomeController(ILocalEnvironment localEnvironment)
    {
        _localEnvironment = localEnvironment;
    }
    
    public IActionResult Index()
    {
        return Ok($"Hello {_localEnvironment.AppUid} {_localEnvironment.AppLocation} {_localEnvironment.Port}");
    }
}