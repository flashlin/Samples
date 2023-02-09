using DemoSite.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using T1.Roslyn;

namespace DemoSite.Controllers
{
	public class RunCodeViewModel
	{
		public string Code { get; set; } = string.Empty;
		public string Result { get; set; } = string.Empty;
	}

	public class HomeController : Controller
	{
		private readonly ILogger<HomeController> _logger;

		public HomeController(ILogger<HomeController> logger)
		{
			_logger = logger;
		}

		public IActionResult Index()
		{
			return View(new RunCodeViewModel());
		}

		public IActionResult RunCode(RunCodeViewModel vm)
		{
			var code = $@"
using System;
namespace MyDynamicNs {{
public class MyDynamicType {{
				public string Execute() {{ 
						{vm.Code}
					}}
				}}
}}";

			var roslyn = new RoslynScripting();
			var compileResult = roslyn.Compile(code);
			compileResult.Match(assembly =>
			{
				dynamic instance = assembly.CreateInstance("MyDynamicNs.MyDynamicType")!;
				vm.Result = $"{instance.Execute()}";
				return vm;
			}, compileError =>
			{
				vm.Result = string.Join("\r\n", compileError.Errors);
				return vm;
			});
			return View("Index", vm);
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