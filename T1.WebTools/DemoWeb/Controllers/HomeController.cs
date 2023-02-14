using DemoWeb.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using T1.WebTools.CsvEx;

namespace DemoWeb.Controllers
{
	public class HomeController : Controller
	{
		private readonly ILogger<HomeController> _logger;

		public HomeController(ILogger<HomeController> logger)
		{
			_logger = logger;
		}

		public IActionResult Index()
		{
			var vm = new IndexViewModel();

			var csvData = @"id,name
1,flash
2,jack
3,mary";

			vm.CsvSheet = CsvSheet.ReadFrom(csvData);
			
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