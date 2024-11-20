namespace SqlSharpLit.Common.ParserLit;

public class SqlConstraintColumn
{
    public string ColumnName { get; set; } = string.Empty;
    public string Order { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"{ColumnName} {Order}";
    }
}