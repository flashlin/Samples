namespace T1.SqlSharp.Expressions;

public class SqlOrderColumn : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OrderColumn;
    public TextSpan Span { get; set; } = new();
    public string ColumnName { get; set; } = string.Empty;
    public OrderType Order { get; set; }

    public string ToSql()
    {
        return $"{ColumnName} {Order.ToString().ToUpper()}";
    }
}