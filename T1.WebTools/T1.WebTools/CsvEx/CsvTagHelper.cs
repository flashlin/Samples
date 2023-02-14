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


public static class CsvHtmlHelper
{
    public static HtmlString CsvHeadersTypeSelect(this IHtmlHelper htmlHelper, 
        string label, string name, List<CsvHeader> selectList)
    {
        var formGroup = new TagBuilder("div");
        formGroup.MergeAttribute("class", "form-group");
        
        foreach (var item in selectList)
        {
            var columnTag = htmlHelper.CsvHeaderTypeSelect(item.Name, item.Name, item);
            formGroup.InnerHtml.AppendHtml(columnTag);
        }

        return new HtmlString(formGroup.ToHtmlString());
    }
    
    public static HtmlString CsvHeaderTypeSelect(this IHtmlHelper htmlHelper, 
        string label, string name, CsvHeader csvHeader)
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

        var columnType = csvHeader.ColumnType;
        var columnTypeList = csvHeader.ColumnType.QueryEnumKeyValues();
        foreach (var item in columnTypeList)
        {
            var optionTag = new TagBuilder("option");
            optionTag.MergeAttribute("value", $"{item.Value}");
            optionTag.InnerHtml.Append(item.Key);
            if (item.Value == columnType)
            {
                optionTag.MergeAttribute("selected", "selected");
            }
            selectTag.InnerHtml.AppendHtml(optionTag);
        }

        formGroup.InnerHtml.AppendHtml(selectTag);
        return new HtmlString(formGroup.ToHtmlString());
    }

    public static IEnumerable<KeyValuePair<string, T>> QueryEnumKeyValues<T>(this T enumInstance)
        where T: Enum
    {
        var names = Enum.GetNames(typeof(T));
        foreach (var name in names)
        {
            var value = (T)Enum.Parse(typeof(T), name);
            yield return new KeyValuePair<string, T>(name, value);
        }
    }

    private static string ToHtmlString(this TagBuilder tagBuilder)
    {
        using var html = new StringWriter();
        tagBuilder.WriteTo(html, System.Text.Encodings.Web.HtmlEncoder.Default);
        return html.ToString();
    }
}