namespace SqlSharpLit.Common.ParserLit;

public class SqlColumnConstraint
{
    public string ColumnName { get; set; } = string.Empty;
    public string Order { get; set; } = "ASC";
    public string ToSql()
    {
        return $"{ColumnName} {Order}";
    }
}