namespace SqlSharpLit.Common.ParserLit;

public class SqlWhereExpression : ISqlWhereExpression 
{
    public ISqlExpression Left { get; set; }
    public string Operation { get; set; } = string.Empty;
    public ISqlExpression Right { get; set; }
}