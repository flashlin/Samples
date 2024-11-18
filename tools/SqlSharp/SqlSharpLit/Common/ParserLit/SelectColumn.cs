namespace SqlSharpLit.Common.ParserLit;

public class SelectColumn : ISelectColumnExpression
{
    public string ColumnName { get; set; } = string.Empty;
    public string ToSql()
    {
        return ColumnName;
    }
}