using System.Text;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharp.DatabaseDescriptions;

public class TableDescription
{
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDescription> Columns { get; set; } = [];
    public string Description { get; set; } = string.Empty;
    
    public string ToDescriptionText()
    {
        var sb = new StringBuilder();
        sb.Append($"Table: {TableName}");
        if (!string.IsNullOrEmpty(Description))
        {
            sb.Append($" -- {Description}");
        }
        sb.AppendLine();
        sb.AppendLine("Columns:");
        foreach (var column in Columns)
        {
            sb.AppendLine(column.ToDescriptionText());
        }
        return sb.ToString();
    }

    public void UpdateColumnsDescription(TableDescription userTable)
    {
        var columns = Columns;
        foreach (var column in columns)
        {
            var userColumn = userTable.Columns.FirstOrDefault(x => x.ColumnName.IsSameAs(column.ColumnName)) ?? new ColumnDescription();
            column.Description = string.IsNullOrEmpty(userColumn.Description) ? column.Description : userColumn.Description;
        }
    }
}