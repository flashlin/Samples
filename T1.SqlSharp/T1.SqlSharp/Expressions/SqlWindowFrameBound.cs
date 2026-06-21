using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWindowFrameBound : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WindowFrameBound;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WindowFrameBound(this);
    }

    public SqlFrameBoundKind Kind { get; set; }
    public ISqlExpression? Offset { get; set; }

    public string ToSql()
    {
        return Kind switch
        {
            SqlFrameBoundKind.UnboundedPreceding => "UNBOUNDED PRECEDING",
            SqlFrameBoundKind.UnboundedFollowing => "UNBOUNDED FOLLOWING",
            SqlFrameBoundKind.CurrentRow => "CURRENT ROW",
            SqlFrameBoundKind.Preceding => $"{Offset?.ToSql()} PRECEDING",
            SqlFrameBoundKind.Following => $"{Offset?.ToSql()} FOLLOWING",
            _ => string.Empty
        };
    }
}
