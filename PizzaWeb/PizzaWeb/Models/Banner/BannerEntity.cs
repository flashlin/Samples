using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using PizzaWeb.Models.Helpers;
using T1.Standard.Common;

namespace PizzaWeb.Models.Banner;

[Table("Banners")]
public class BannerEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }

    public string Name { get; set; } = "";
    public int OrderId { get; set; } = 1;
    public string VariableOptions { get; set; } = "{}"; // { name: resxName } 
    public string TemplateName { get; set; } = "";
    public DateTime LastModifiedTime { get; set; } = DateTime.UtcNow;
}

public class Banner
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public int OrderId { get; set; } = 1;
    public List<BannerVariable> Variables { get; set; } = new List<BannerVariable>();
    public string TemplateName { get; set; } = "";
    public DateTime LastModifiedTime { get; set; } = DateTime.UtcNow;

    public static List<TemplateVariableValue> ParseVariableOptionsJson(string variableOptions)
    {
        var dict = new Dictionary<string, TemplateVariableValue>();
        if (!string.IsNullOrEmpty(variableOptions))
        {
            var sp = ServiceLocator.Current;
            var jsonConvert = sp.GetService<IJsonConverter>();
            dict = jsonConvert.Deserialize<Dictionary<string, TemplateVariableValue>>(variableOptions);
        }
        return dict.Values.ToList();
    }
}

public class BannerVariable
{
    public string VarName { get; set; }
    public string ResxName { get; set; }
    public List<VariableResx> ResxList { get; set; } = new List<VariableResx>();
}

public class VariableResx
{
    public string IsoLangCode { get; set; }
    public string Content { get; set; }
}

public class TemplateVariableValue
{
    public string VarName { get; set; }
    public string ResxName { get; set; }
}