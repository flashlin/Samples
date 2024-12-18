using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForXmlRootDirective : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ForXmlRootDirective;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ForXmlRootDirective(this);
    }

    public ISqlExpression? RootName { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("ROOT");
        if (RootName != null)
        {
            sql.Write("(");
            sql.Write(RootName.ToSql());
            sql.Write(")");
        }
        return sql.ToString();
    }
}