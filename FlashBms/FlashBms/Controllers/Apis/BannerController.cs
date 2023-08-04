using FlashBms.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.Mvc.Razor;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using RazorLight;
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

    [HttpGet]
    public async Task<string> RenderCshtml()
    {
        var template = @"<p>Hello @Model.Name</p>";
        var model = new {Name = "Flash"};
        var engine = new RazorRenderService();
        return await engine.CompileRenderStringAsync("templateKey", template, model);
    }
}

public interface IRazorRenderService
{
    Task<string> CompileRenderStringAsync<TModel>(string templateKey, string content, TModel model);
}

public class RazorRenderService : IRazorRenderService
{
    private readonly RazorLightEngine _engine = new RazorLightEngineBuilder()
        .UseEmbeddedResourcesProject(typeof(RazorRenderService))
        .SetOperatingAssembly(typeof(RazorRenderService).Assembly)
        .UseMemoryCachingProvider()
        .Build();

    public Task<string> CompileRenderStringAsync<TModel>(string templateKey, string content, TModel model)
    {
        return _engine.CompileRenderStringAsync(templateKey, content, model);
    }
}