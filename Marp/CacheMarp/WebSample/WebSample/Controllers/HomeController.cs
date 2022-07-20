using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebSample.Models;
using WebSample.Services;

namespace WebSample.Controllers
{
	public class HomeController : Controller
	{
		private readonly ILogger<HomeController> _logger;
		private readonly IUserService _userService;
		private readonly IGlobalSettingService _globalSettingService;
		private readonly MyGlobalSettings _myGlobalSettings;

		public HomeController(ILogger<HomeController> logger, 
			IUserService userService,
			IGlobalSettingService globalSettingService,
			IGlobalSettingFactory<MyGlobalSettings> globalSettingFactory)
		{
			_globalSettingService = globalSettingService;
			_userService = userService;
			_logger = logger;
			_myGlobalSettings = globalSettingFactory.Create();
		}

		public IActionResult Index()
		{
			var featureEnabled1 = _globalSettingService.GetStringValue("FeatureEnabled") == "true";
			var featureEnabled2 = _globalSettingService.GetBoolValue("FeatureEnabled");
			var featureEnabled3 = _myGlobalSettings.FeatureEnabled;
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