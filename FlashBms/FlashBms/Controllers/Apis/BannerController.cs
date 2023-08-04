using FlashBms.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using T1.AspNetCore;

namespace FlashBms.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class BannerController : ControllerBase
{
    private IViewToStringRendererService _viewToStringRendererService;

    public BannerController(IViewToStringRendererService viewToStringRendererService)
    {
        _viewToStringRendererService = viewToStringRendererService;
    }
    
    [HttpPost]
    public async Task<string> Render()
    {
        var viewModel = new SampleViewModel();
        return await _viewToStringRendererService.RenderViewToStringAsync("~/Views/Banners/Sample.cshtml", viewModel);
    }
}