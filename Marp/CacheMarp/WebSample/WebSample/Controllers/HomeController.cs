using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebSample.Models;
using WebSample.Services;

namespace WebSample.Controllers
{
	public class HomeController : Controller
	{
		private readonly ILogger<HomeController> _logger;
		private IUserService _userService;

		public HomeController(ILogger<HomeController> logger, IUserService userService)
		{
			_userService = userService;
			_logger = logger;
		}

		public IActionResult Index()
		{
			var vm = new IndexViewModel()
			{
				User = _userService.GetUser("flash")
			};
			return View(vm);
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
}