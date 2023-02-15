using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;

namespace QueryApp.Controllers;

public class HomeController : Controller
{
    private IClientEnvironment _clientEnvironment;

    public HomeController(IClientEnvironment clientEnvironment)
    {
        _clientEnvironment = clientEnvironment;
    }
    
    public IActionResult Index()
    {
        return Ok("Hello " + _clientEnvironment.Port);
    }
}