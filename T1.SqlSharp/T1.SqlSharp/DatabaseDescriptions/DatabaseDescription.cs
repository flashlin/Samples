using System.Text;

namespace T1.SqlSharp.DatabaseDescriptions;

public class DatabaseDescription
{
    public string DatabaseName { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public List<TableDescription> Tables { get; set; } = [];
    public string ToDescriptionText()
    {
        var sb = new StringBuilder();
        sb.Append($"Database: {DatabaseName}");
        if(!string.IsNullOrEmpty(Description))
        {
            sb.Append($" -- {Description}");
        }
        sb.AppendLine();
        foreach (var table in Tables)
        {
            sb.AppendLine(table.ToDescriptionText());
            sb.AppendLine("---");
            sb.AppendLine();
            sb.AppendLine();
        }
        return sb.ToString();
    }
}