using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using T1.SourceGenerator.Attributes;
using WebDemoSite.Models;
using T1.SourceGenerator;
namespace WebDemoSite.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;

    public HomeController(ILogger<HomeController> logger)
    {
        _logger = logger;
    }

    public IActionResult Index()
    {
        var a = new UserEntity()
        {
            Name = "flash"
        };
        var b = new UserDto();
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


[AutoMapping(typeof(UserEntity), typeof(UserDto))]
public class UserEntity
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
}

public class UserDto
{
    public string Name { get; set; } = string.Empty;
}