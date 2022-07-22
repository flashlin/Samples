using Microsoft.AspNetCore.Mvc;
using WebDemo.Models;
using WebDemo.Services;

namespace WebDemo.Controllers
{
	public class AsyncInCtlController : Controller
	{
		private readonly IWeatherService _weatherService;

		public AsyncInCtlController(IWeatherService weatherService)
		{
			_weatherService = weatherService;
		}

		public IActionResult Index()
		{
			var vm = new AsyncInCtlModel
			{
				Info = _weatherService.GetInfoAsync("Taipei").Result
			};
			return View(vm);
		}
	}
}
