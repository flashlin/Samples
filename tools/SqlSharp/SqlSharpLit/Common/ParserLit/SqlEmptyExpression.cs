namespace SqlSharpLit.Common.ParserLit;

public class SqlEmptyExpression : ISqlExpression
{
    public string ToSql()
    {
        return string.Empty;
    }
}