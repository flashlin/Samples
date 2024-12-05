namespace T1.SqlSharp.Expressions;

public class SqlConditionExpression : ISqlWhereExpression 
{
    public SqlType SqlType { get; } = SqlType.Condition;
    public required ISqlExpression Left { get; set; }
    public ComparisonOperator ComparisonOperator { get; set; }

    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} {ComparisonOperator.ToString()} {Right.ToSql()}";
    }
}