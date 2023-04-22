using System.Text;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.Routing;
using Microsoft.AspNetCore.Mvc.TagHelpers;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Razor.TagHelpers;

namespace QueryWeb.Models.TagHelpers;

[HtmlTargetElement(TagHelperName, Attributes = HrefAttributeName, TagStructure = TagStructure.WithoutEndTag)]
public class CustomLinkTagHelper : TagHelper
{
    private const string TagHelperName = "link";
    private const string RelAttributeName = "rel";
    private const string HrefAttributeName = "href";
    private IUrlHelperFactory _urlHelperFactory;
    private IPathBaseFeature _pathBaseFeature;

    public CustomLinkTagHelper(
        IPathBaseFeature pathBaseFeature,
        IUrlHelperFactory urlHelperFactory)
    {
        _pathBaseFeature = pathBaseFeature;
        _urlHelperFactory = urlHelperFactory;
    }

    [HtmlAttributeNotBound] [ViewContext] public ViewContext ViewContext { get; set; } = null!;

    [HtmlAttributeName(HrefAttributeName)] public string Href { get; set; } = string.Empty;

    [HtmlAttributeName(RelAttributeName)] public string Rel { get; set; } = string.Empty;

    public override Task ProcessAsync(TagHelperContext context, TagHelperOutput output)
    {
        var href = Href.Replace("~/", _pathBaseFeature.PathBase + "/");
        var urlHelper = _urlHelperFactory.GetUrlHelper(ViewContext);
        output.Attributes.SetAttribute("href", urlHelper.Content(href));
        output.Attributes.SetAttribute(RelAttributeName, Rel);
        return Task.CompletedTask;
    }

    private T GetRequiredService<T>()
    {
        return ViewContext.HttpContext.RequestServices.GetRequiredService<T>();
    }
}