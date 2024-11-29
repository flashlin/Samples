using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlWhereExpression : ISqlWhereExpression 
{
    public required ISqlExpression Left { get; set; }
    public string Operation { get; set; } = string.Empty;
    public required ISqlExpression Right { get; set; }
    public string ToSql()
    {
        return $"{Left.ToSql()} {Operation} {Right.ToSql()}";
    }
}