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
            ComparisonOperator.Is => "IS",
            ComparisonOperator.IsNot => "IS NOT",
            ComparisonOperator.NotLike=> "NOT LIKE",
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
            "IS" => ComparisonOperator.Is,
            "IS NOT" => ComparisonOperator.IsNot,
            "NOT LIKE" => ComparisonOperator.NotLike,
            _ => throw new NotImplementedException()
        };
    }
}