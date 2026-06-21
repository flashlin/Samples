namespace T1.SqlSharp.Expressions;

public class SqlOverWindowName : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OverWindowName;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OverWindowName(this);
    }

    public ISqlExpression Field { get; set; } = new SqlFieldExpr();
    public string WindowName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{Field.ToSql()} OVER {WindowName}";
    }
}
