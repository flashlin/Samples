using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text;
using PizzaWeb.Models.Helpers;
using T1.Standard.Common;

namespace PizzaWeb.Models.Banner;

[Table("Banner")]
public class BannerEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }

    public string BannerName { get; set; } = String.Empty;
    public int OrderId { get; set; }
    public List<VariableOption> VariableOptions { get; set; } = new List<VariableOption>();
    public string TemplateName { get; set; } = String.Empty;
    public DateTime LastModifiedTime { get; set; } = DateTime.UtcNow;
}

public class BannerSetting
{
	public int Id { get; set; }
    public string TemplateName { get; set; } = String.Empty;
    public string BannerName { get; set; } = String.Empty;
    public int OrderId { get; set; }
    public List<BannerVariable> Variables { get; set; } = new List<BannerVariable>();
    public DateTime LastModifiedTime { get; set; } = DateTime.UtcNow;
}

public class BannerVariable
{
    public string VarName { get; set; } = string.Empty;
    public string ResxName { get; set; } = string.Empty;
    public List<VariableResx> ResxList { get; set; } = new List<VariableResx>();
	public override string ToString()
	{
		var sb = new StringBuilder();
      sb.Append(VarName);
		sb.Append("=");
      sb.Append(ResxName);
		sb.Append(" ");
		sb.Append(string.Join(",", ResxList.Select(x => x.ToString())));
		return sb.ToString();
	}
}

public class VariableResx
{
    public string IsoLangCode { get; set; } = "en-US";
    public string Content { get; set; } = string.Empty;
	public override string ToString()
	{
		return $"{IsoLangCode}:'{Content}'";
	}
}

public class VariableOption
{
    public string VarName { get; set; } = string.Empty;
    public string ResxName { get; set; } = string.Empty;
}