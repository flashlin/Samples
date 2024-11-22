namespace SqlSharpLit.Common.ParserLit;

public class SqlNoneExpression : ISqlExpression
{
    public static SqlNoneExpression Default { get; } = new();
    public SqlType SqlType => SqlType.None;
    public string ToSql()
    {
        return string.Empty;
    }
}