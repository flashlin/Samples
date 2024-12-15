using T1.SqlSharp;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlToken : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Token;
    public string Value { get; set; } = string.Empty;
    public required TextSpan Span { get; set; }
    public string ToSql()
    {
        return Value;
    }

}