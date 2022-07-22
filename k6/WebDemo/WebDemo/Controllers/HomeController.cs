using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using NuGet.Common;
using WebDemo.Models;
using ILogger = Microsoft.Extensions.Logging.ILogger;
using LogLevel = Microsoft.Extensions.Logging.LogLevel;

namespace WebDemo.Controllers
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
			_logger.LogMyMessage(100);
			return UsingAsyncInView();
		}

		public IActionResult UsingAsyncInView()
		{
			return View("UsingAsyncInView");
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

	public static class LoggerMessageDefinitions
	{
		private static readonly Action<ILogger, int, Exception?> LogMessageDefinition =
			LoggerMessage.Define<int>(LogLevel.Information, new EventId(0), 
				"message {errorCode}");

		public static void LogMyMessage(this ILogger logger, int errorCode)
		{
			LogMessageDefinition(logger, errorCode, null);
		}
	}


	public static partial class LoggerMessageDefinitionsGen
	{
		[LoggerMessage(EventId = 0, 
			Level = LogLevel.Information, 
			Message = "message {errorCode}",
			SkipEnabledCheck = true)]
		public static partial void LogMyMessageGen(this ILogger logger, int errorCode);
	}
}