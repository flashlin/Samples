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
    public string Escape { get; set; } = string.Empty;

    public string ToSql()
    {
        var escape = string.IsNullOrEmpty(Escape) ? string.Empty : $" ESCAPE {Escape}";
        return $"{Left.ToSql()} {ComparisonOperator.ToSql()} {Right.ToSql()}{escape}";
    }
}