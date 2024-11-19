namespace SqlSharpLit.Common.ParserLit;

public class SqlParameterValue : ISqlExpression
{
    public SqlType SqlType => SqlType.ParameterValue;
    public string Name { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $@"{Name}={Value}";
    }
}