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

    public bool IsAll { get; set; } = false;
    public required ISqlExpression SelectStatement { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("UNION ");
        if (IsAll)
        {
            sql.Write("ALL ");
        }
        sql.Write(SelectStatement.ToSql());
        return sql.ToString();
    }
}