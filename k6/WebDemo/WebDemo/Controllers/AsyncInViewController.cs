using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebDemo.Models;

namespace WebDemo.Controllers
{
	public class AsyncInViewController : Controller
	{
		public AsyncInViewController()
		{
		}

		public IActionResult Index()
		{
			return View();
		}
	}
}