using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlUnionSelect : ISqlExpression
{
    public SqlType SqlType => SqlType.UnionSelect;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_UnionSelect(this);
    }

    public SqlSetOperator Operator { get; set; } = SqlSetOperator.Union;
    public bool IsAll => Operator == SqlSetOperator.UnionAll;
    public required ISqlExpression SelectStatement { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write($"{GetOperatorKeyword()} ");
        sql.Write(SelectStatement.ToSql());
        return sql.ToString();
    }

    private string GetOperatorKeyword()
    {
        return Operator switch
        {
            SqlSetOperator.UnionAll => "UNION ALL",
            SqlSetOperator.Intersect => "INTERSECT",
            SqlSetOperator.Except => "EXCEPT",
            _ => "UNION"
        };
    }
}