using Microsoft.AspNetCore.Mvc.Routing;

namespace MockApiWeb.Models.Middlewares;

public class DynamicApiTransformer : DynamicRouteValueTransformer
{
    public override ValueTask<RouteValueDictionary> TransformAsync(HttpContext httpContext, RouteValueDictionary values)
    {
        if (!values.ContainsKey("controller") || !values.ContainsKey("action"))
        {
            return ValueTask.FromResult(values);
        }
 
        var controller = (string?)values["controller"];
        if (controller == null) return ValueTask.FromResult(values);
        values["controller"] = "MockWebApi";
 
        var action = (string?)values["action"];
        if (action == null) return ValueTask.FromResult(values);
        values["action"] = "ProcessRequest";
 
        return ValueTask.FromResult(values);
    }
}