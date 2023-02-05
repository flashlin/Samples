using Microsoft.AspNetCore.Mvc;

namespace T1.WebTools.Controllers;

public class ToolsController : Controller
{
	public IActionResult Index()
	{
		return Ok("Message from shared assembly!");
	}

	public IActionResult IndexView()
	{
		// This method requires 
		// .UseStartup<StartupViews>();
		// in Program.cs
		return View("Index");
	}
}