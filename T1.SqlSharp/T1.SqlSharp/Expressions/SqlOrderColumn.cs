namespace T1.SqlSharp.Expressions;

public class SqlOrderColumn : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OrderColumn;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OrderColumn(this);
    }

    public required ISqlExpression ColumnName { get; set; }
    public OrderType Order { get; set; }

    public string ToSql()
    {
        return $"{ColumnName.ToSql()} {Order.ToString().ToUpper()}";
    }
}