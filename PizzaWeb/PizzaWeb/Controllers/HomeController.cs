using Microsoft.AspNetCore.Mvc;
using PizzaWeb.Models;
using System.Diagnostics;

namespace PizzaWeb.Controllers
{
	public class IndexViewModel
	{
		public List<StoreShelvesEntity> StoreShelves { get; set; }
	}

	public class HomeController : Controller
	{
		private readonly ILogger<HomeController> _logger;
		private readonly PizzaDbContext _dbContext;

		public HomeController(ILogger<HomeController> logger, PizzaDbContext dbContext)
		{
			_logger = logger;
			this._dbContext = dbContext;
		}

		public IActionResult Index()
		{
			return View(new IndexViewModel
			{
				StoreShelves = _dbContext.StoreShelves.ToList()
			});
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