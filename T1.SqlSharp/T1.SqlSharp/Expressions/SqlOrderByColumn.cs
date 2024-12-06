namespace T1.SqlSharp.Expressions;

public class SqlOrderByColumn : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OrderByColumn;
    public string ColumnName { get; set; } = string.Empty;
    public OrderType Order { get; set; }

    public string ToSql()
    {
        return $"{ColumnName} {Order.ToString().ToUpper()}";
    }
}