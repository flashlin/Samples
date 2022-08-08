using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace WCodeSnippetX.Controllers.api
{
	[Route("/api/[controller]/[action]")]
	public class CodeSnippetController : ControllerBase
	{
		[HttpGet]
		public string SayHello()
		{
			return "Say Hello";
		}
	}
}
