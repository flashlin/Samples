using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlValue : ISqlValue, ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.String;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{Value}";
    }
}