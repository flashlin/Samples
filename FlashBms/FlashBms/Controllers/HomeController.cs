using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using FlashBms.Models;
using T1.AspNetCore;

namespace FlashBms.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private IViewToStringRendererService _viewToStringRendererService;

    public HomeController(ILogger<HomeController> logger,
        IViewToStringRendererService viewToStringRendererService)
    {
        _viewToStringRendererService = viewToStringRendererService;
        _logger = logger;
    }

    public async Task<IActionResult> Index()
    {
        var viewModel = new BannerViewModel();
        viewModel.Banners.Add(new DesignBanner());
        var htmlContent = await _viewToStringRendererService.RenderViewToStringAsync("~/Views/Banners/Sample.cshtml", viewModel);

        var banner = viewModel.Banners[0] as DesignBanner;
        var names = banner!.Settings.Select(x => x.Name).ToList();

        return Content(htmlContent, "text/html");
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