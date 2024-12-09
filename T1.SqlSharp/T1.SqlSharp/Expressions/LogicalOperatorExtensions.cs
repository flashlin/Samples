namespace T1.SqlSharp.Expressions;

public static class LogicalOperatorExtensions
{
    public static string ToSql(this LogicalOperator logicalOperator)
    {
        return logicalOperator switch
        {
            LogicalOperator.And => "AND",
            LogicalOperator.Or => "OR",
            LogicalOperator.Not => "NOT",
            _ => string.Empty
        };
    }
}