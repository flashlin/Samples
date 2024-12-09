namespace T1.SqlSharp.Expressions;

public static class ComparisonOperatorExtensions
{
    public static string ToSql(this ComparisonOperator comparisonOperator)
    {
        return comparisonOperator switch
        {
            ComparisonOperator.Equal => "=",
            ComparisonOperator.NotEqual => "!=",
            ComparisonOperator.GreaterThan => ">",
            ComparisonOperator.LessThan => "<",
            ComparisonOperator.GreaterThanOrEqual => ">=",
            ComparisonOperator.LessThanOrEqual => "<=",
            ComparisonOperator.Like => "LIKE",
            ComparisonOperator.In => "IN",
            ComparisonOperator.Between => "BETWEEN",
            ComparisonOperator.IsNull => "IS NULL",
            ComparisonOperator.IsNot => "IS NOT NULL",
            _ => throw new NotImplementedException()
        };
    }
}