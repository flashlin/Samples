using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using FrameworkWebSite.Services;
using FrameworkWebSite.ViewModels;

namespace FrameworkWebSite.Controllers
{
    public class HomeController : Controller
    {
        private IUserService _userService;

        public HomeController(IUserService userService)
        {
            _userService = userService;
        }
        
        public ActionResult Index()
        {
            var vm = new IndexViewModel()
            {
                User = _userService.GetUser("flash")
            };
            return View(vm);
        }

        public ActionResult About()
        {
            ViewBag.Message = "Your application description page.";
            return View();
        }

        public ActionResult Contact()
        {
            ViewBag.Message = "Your contact page.";
            return View();
        }
    }
}