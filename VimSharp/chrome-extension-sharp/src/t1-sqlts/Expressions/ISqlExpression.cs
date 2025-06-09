namespace T1.SqlSharp.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
    TextSpan Span { get; set; }
    void Accept(SqlVisitor visitor);
}

public class SqlExpressionNode
{
    public required ISqlExpression Expression { get; set; }
    public int Depth { get; set; }
}