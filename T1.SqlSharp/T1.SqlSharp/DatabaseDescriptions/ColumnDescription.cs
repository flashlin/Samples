using System.Text;

namespace T1.SqlSharp.DatabaseDescriptions;

public class ColumnDescription
{
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public bool IsNullable { get; set; }
    public bool IsIdentity { get; set; }
    public string DefaultValue { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;

    public string ToDescriptionText()
    {
        var text = new StringBuilder();
        text.Append($"{ColumnName} {DataType}");
        if (IsNullable)
        {
            text.Append($",Is Nullable");
        }
        if (IsIdentity)
        {
            text.Append($",Is Identity");
        }
        if (!string.IsNullOrEmpty(DefaultValue))
        {
            text.Append($",Default: {DefaultValue}");
        }
        if (!string.IsNullOrEmpty(Description))
        {
            text.Append($"-- {Description}");
        }
        return text.ToString();
    }
}