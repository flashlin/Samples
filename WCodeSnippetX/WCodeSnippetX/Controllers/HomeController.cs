using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace WCodeSnippetX.Controllers
{
	[Route("/[controller]/[action]")]
	public class HomeController : Controller
	{
		public IActionResult Index()
		{
			return View("index");
		}
	}
}
