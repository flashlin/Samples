using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlToken : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Token;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return Value;
    }
}