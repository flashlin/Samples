using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlSearchCondition : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.SearchCondition;
    public required ISqlExpression Left { get; set; }
    public LogicalOperator LogicalOperator { get; set; } = LogicalOperator.None;
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