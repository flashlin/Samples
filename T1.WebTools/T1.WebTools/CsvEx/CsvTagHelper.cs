using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Razor.TagHelpers;

namespace T1.WebTools.CsvEx;


[HtmlTargetElement("CsvHeadersTypeSelect")]
public class CsvHeadersTypeSelectTagHelper : TagHelper
{
    private readonly IHtmlHelper _htmlHelper;

    public CsvHeadersTypeSelectTagHelper(IHtmlHelper htmlHelper)
    {
        _htmlHelper = htmlHelper;
    }
    
    [HtmlAttributeName("Label")] 
    public string Label { get; set; } = string.Empty;
    
    
    [HtmlAttributeName("Name")] 
    public string Name { get; set; } = string.Empty;
    
    [HtmlAttributeName("Items")] 
    public List<CsvHeader> Items { get; set; } = new();
    
    public override void Process(TagHelperContext context, TagHelperOutput output)
    {
        // output.TagName = "div";
        // output.Attributes.SetAttribute("class", "form-group");
        // output.Content.SetHtmlContent("This is a custom tag");
        //childContent.AppendHtml(selectTag);
        //var childContent = output.GetChildContentAsync().Result;

        var selectTag = _htmlHelper.CsvHeadersTypeSelect(Label, Name, Items);
        output.Content.AppendHtml(selectTag);
    }
}