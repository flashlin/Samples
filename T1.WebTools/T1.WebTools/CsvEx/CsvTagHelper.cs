using Microsoft.AspNetCore.Html;
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
    
    public override void Process(TagHelperContext context, TagHelperOutput output)
    {
        // output.TagName = "div";
        // output.Attributes.SetAttribute("class", "form-group");
        // output.Content.SetHtmlContent("This is a custom tag");
        var childContent = output.GetChildContentAsync().Result;

        var selectTag = _htmlHelper.CsvHeadersTypeSelect("", "", null);
        output.Content.AppendHtml(selectTag);
    }
}


public static class CsvHtmlHelper
{
    public static HtmlString CsvHeadersTypeSelect(this IHtmlHelper htmlHelper, 
        string label, string name, List<CsvHeader> selectList)
    {
        var formGroup = new TagBuilder("div");
        formGroup.MergeAttribute("class", "form-group");
        
        var labelTag = new TagBuilder("label");
        labelTag.MergeAttribute("for", name);
        labelTag.InnerHtml.Append(label);
        formGroup.InnerHtml.AppendHtml(labelTag);
        
        var selectTag = new TagBuilder("select");
        selectTag.MergeAttribute("name", name);
        selectTag.MergeAttribute("id", name);
        selectTag.MergeAttribute("class", "form-control");

        foreach (var item in selectList)
        {
            var optionTag = new TagBuilder("option");
            optionTag.MergeAttribute("value", $"{item.ColumnType}");
            optionTag.InnerHtml.Append(item.Name);
            selectTag.InnerHtml.AppendHtml(optionTag);
        }

        formGroup.InnerHtml.AppendHtml(selectTag);
        return new HtmlString(formGroup.ToString());
    }
}