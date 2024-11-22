namespace SqlSharpLit.Common.ParserLit;

public class SqlStringValue : ISqlValue, ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.String;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{Value}";
    }
}