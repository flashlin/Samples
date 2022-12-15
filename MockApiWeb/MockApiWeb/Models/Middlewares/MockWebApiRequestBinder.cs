using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using MockApiWeb.Models.Requests;

namespace MockApiWeb.Models.Middlewares;

public class MockWebApiRequestBinder : IModelBinder
{
    public async Task BindModelAsync(ModelBindingContext bindingContext)
    {
        var reqPath = bindingContext.HttpContext.Request.Path.Value!;
        var identifierPattern = "[A-Za-z_][A-Za-z_\\d]+";
        var rg = new Regex(
            $"^/mock_(?<product>{identifierPattern})/api/(?<controller>{identifierPattern})/(?<action>{identifierPattern})");
        var match = rg.Match(reqPath);
        if (!match.Success)
        {
            return;
        }

        using var reader = new StreamReader(bindingContext.ActionContext.HttpContext.Request.Body);
        var requestBody = await reader.ReadToEndAsync();

        bindingContext.Result = ModelBindingResult.Success(new MockWebApiRequest()
        {
            ProductName = match.Groups["product"].Value,
            ControllerName = match.Groups["controller"].Value,
            ActionName = match.Groups["action"].Value,
            RequestBody = requestBody,
            RequestQueryString = bindingContext.ActionContext.HttpContext.Request.QueryString.Value ?? string.Empty
        });
        //bindingContext.ModelState.SetModelValue("ProductName", ModelBindingResult.Success(reqPath));
        // var modelName = bindingContext.ModelName;
        // var valueProviderResult = bindingContext.ValueProvider.GetValue(modelName);
        // if (valueProviderResult == ValueProviderResult.None)
        // {
        //     return Task.CompletedTask;
        // }
        // bindingContext.ModelState.SetModelValue(modelName, valueProviderResult);
        // bindingContext.Result = ModelBindingResult.Success(model);
    }
}