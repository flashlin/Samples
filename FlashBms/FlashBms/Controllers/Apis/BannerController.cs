using FlashBms.Models.ViewModels;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.Mvc.Razor;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
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

    /*
    public void RenderCshtml()
    {
        var cshtml = @"<p>Hello @Model.Name</p>";
        var model = new { Name = "Flash" };
        return RenderViewContextToStringAsync(cshtml, model);
    }

    private async Task<string> RenderViewContextToStringAsync<TModel>(string viewContent, TModel model)
    {
        var viewEngine = new RazorViewEngine();
        var actionContext = new ActionContext
        {
            HttpContext = new DefaultHttpContext(),
        };
        var view = viewEngine.GetView(null, viewContent, true).View;
        var viewData = new ViewDataDictionary<TModel>(new EmptyModelMetadataProvider(), new ModelStateDictionary())
        {
            Model = model
        };
        var tempData = new TempDataDictionary(actionContext.HttpContext, new SessionStateTempDataProvider());
        var output = new StringWriter();
        var viewContext = new ViewContext(
            actionContext,
            view,
            viewData,
            tempData,
            output,
            new HtmlHelperOptions()
        );

        await view.RenderAsync(viewContext);
        return output.ToString();
    }
    */
}