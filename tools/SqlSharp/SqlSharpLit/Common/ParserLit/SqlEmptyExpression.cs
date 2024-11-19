namespace SqlSharpLit.Common.ParserLit;

public class SqlEmptyExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.Empty;
    public string ToSql()
    {
        return string.Empty;
    }
}