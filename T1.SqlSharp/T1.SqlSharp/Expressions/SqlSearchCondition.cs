using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlSearchCondition : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.SearchCondition;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SearchCondition(this);
    }

    public required ISqlExpression Left { get; set; }
    public LogicalOperator LogicalOperator { get; set; } = LogicalOperator.None;
    public TextSpan OperatorSpan { get; set; } = new();
    public ISqlExpression? Right { get; set; }
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Left.ToSql());
        if (LogicalOperator != LogicalOperator.None)
        {
            sql.Write(" ");
            sql.Write(LogicalOperator.ToSql());
            sql.Write(" ");
            sql.Write(Right!.ToSql());
        }
        return sql.ToString();
    }
}