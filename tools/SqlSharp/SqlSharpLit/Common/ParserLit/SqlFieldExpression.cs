namespace SqlSharpLit.Common.ParserLit;

public class SqlFieldExpression : ISqlExpression
{
    public string FieldName { get; set; } = string.Empty;
    public string ToSql()
    {
        return FieldName;
    }
}