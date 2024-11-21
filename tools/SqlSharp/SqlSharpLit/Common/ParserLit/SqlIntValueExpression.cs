namespace SqlSharpLit.Common.ParserLit;

public interface ISqlValue 
{
    string Value { get; }
} 

public class SqlIntValueExpression : ISqlValue, ISqlExpression
{
    public SqlType SqlType => SqlType.IntValue;
    public string Value { get; set; } = string.Empty; 
    public string ToSql()
    {
        return $"{Value}";
    }
}