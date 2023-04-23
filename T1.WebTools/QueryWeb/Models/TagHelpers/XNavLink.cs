using System.Globalization;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Rendering;
using Microsoft.AspNetCore.Components.Routing;
using T1.Standard.Collections.Generics;

namespace QueryWeb.Models.TagHelpers;

public class XNavLink : NavLink
{
    [Parameter]
    public new string Href { get; set; }

    [Inject] private IServiceProvider ServiceProvider { get; set; } = null!;


    protected override void BuildRenderTree(RenderTreeBuilder builder)
    {
        var pathBaseFeature = ServiceProvider.GetRequiredService<IPathBaseFeature>();
        
        builder.OpenElement(0, "a");
        builder.AddAttribute(1, "href", pathBaseFeature.GetPath(Href));
        builder.AddMultipleAttributes(2, AdditionalAttributes);
        builder.AddContent(3, ChildContent);
        builder.CloseElement();
    }
}