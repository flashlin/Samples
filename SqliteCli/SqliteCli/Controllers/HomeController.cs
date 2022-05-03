using Microsoft.AspNetCore.Mvc;

namespace SqliteCli.Controllers;

public class HomeController : Controller
{
    public IActionResult Index()
    {
        return Content("Hello World");
    }
}