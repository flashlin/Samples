namespace T1.SqlSharp.Expressions;

public class SqlConditionExpression : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.ComparisonCondition;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ConditionExpression(this);
    }

    public required ISqlExpression Left { get; set; }
    public ComparisonOperator ComparisonOperator { get; set; }
    public TextSpan OperatorSpan { get; set; } = new();
    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} {ComparisonOperator.ToSql()} {Right.ToSql()}";
    }
}