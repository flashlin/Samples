namespace SqlSharpLit.Common.ParserLit;

public class SqlIntValueExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.IntValue;
    public int Value { get; set; }
    public string ToSql()
    {
        return $"{Value}";
    }
}