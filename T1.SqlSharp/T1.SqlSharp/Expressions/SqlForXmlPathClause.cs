using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public interface ISqlForXmlClause : ISqlExpression
{
    List<SqlForXmlRootDirective> CommonDirectives { get; set; }
}

public class SqlForXmlPathClause : ISqlForXmlClause
{
    public SqlType SqlType { get; } = SqlType.ForXmlPathClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ForXmlPathClause(this);
    }

    public string? PathName { get; set; }
    public List<SqlForXmlRootDirective> CommonDirectives { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("FOR XML PATH");
        if (PathName != null)
        {
            sql.Write("(");
            sql.Write(PathName);
            sql.Write(")");
        }
        if(CommonDirectives.Count>0)
        {
            sql.Write(",");
            sql.Write(string.Join(",", CommonDirectives.Select(x => x.ToSql())));
        }
        return sql.ToString();
    }
}