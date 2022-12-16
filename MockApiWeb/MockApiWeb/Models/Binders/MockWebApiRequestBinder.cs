using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using MockApiWeb.Models.DataConstraints;

namespace MockApiWeb.Models.Binders;

public class MockWebApiRequestBinder : IModelBinder
{
    private const string IdentifierPattern = "[A-Za-z_][A-Za-z_\\d]+";
    private readonly Regex[] _requestUrlRegexList = new[]
    {
        $"^/mock_(?<product>{IdentifierPattern})/api/(?<controller>{IdentifierPattern})/(?<action>{IdentifierPattern})",
        $"^/mock_(?<product>{IdentifierPattern})/(?<controller>{IdentifierPattern})/(?<action>{IdentifierPattern})",
    }.Select(x => new Regex(x)).ToArray();

    public async Task BindModelAsync(ModelBindingContext bindingContext)
    {
        var reqPath = bindingContext.HttpContext.Request.Path.Value!;
        var match = Match(reqPath);
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
    }

    private Match Match(string reqPath)
    {
        var len = _requestUrlRegexList.Length;
        for (var n = 0; n < len - 1; n++)
        {
            var match = _requestUrlRegexList[n].Match(reqPath);
            if (match.Success)
            {
                return match;
            }
        }
        return _requestUrlRegexList[len - 1].Match(reqPath);
    }
}