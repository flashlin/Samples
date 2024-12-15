using YamlDotNet.Core.Tokens;

namespace T1.SqlSharp.Expressions;

public class SqlValue : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.String;
    public string Value { get; set; } = string.Empty;
    public TextSpan Span { get; set; } = new();
    public string ToSql()
    {
        return $"{Value}";
    }
}