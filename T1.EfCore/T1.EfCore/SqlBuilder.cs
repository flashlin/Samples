namespace T1.EfCore;

public class SqlBuilder
{
    public string CreateColumns(string tableName, List<SqlRawProperty> rowProperties)
    {
        return string.Join(", ", rowProperties.Select(x => $"[{tableName}].[{x.ColumnName}]"));
    }
}