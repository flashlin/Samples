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
    
    public static ComparisonOperator ToComparisonOperator(this string comparisonOperator)
    {
        return comparisonOperator switch
        {
            "=" => ComparisonOperator.Equal,
            "<>" => ComparisonOperator.NotEqual,
            "!=" => ComparisonOperator.NotEqual,
            ">" => ComparisonOperator.GreaterThan,
            "<" => ComparisonOperator.LessThan,
            ">=" => ComparisonOperator.GreaterThanOrEqual,
            "<=" => ComparisonOperator.LessThanOrEqual,
            "LIKE" => ComparisonOperator.Like,
            "IN" => ComparisonOperator.In,
            "BETWEEN" => ComparisonOperator.Between,
            "IS NULL" => ComparisonOperator.IsNull,
            "IS NOT" => ComparisonOperator.IsNot,
            _ => throw new NotImplementedException()
        };
    }
}