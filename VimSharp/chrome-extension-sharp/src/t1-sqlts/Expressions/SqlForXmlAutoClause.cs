using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForXmlAutoClause : ISqlForXmlClause
{
    public SqlType SqlType { get; } = SqlType.ForXmlAutoClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ForXmlAutoClause(this);
    }

    public List<SqlForXmlRootDirective> CommonDirectives { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("FOR XML AUTO");
        if(CommonDirectives.Count>0)
        {
            sql.Write(" ");
            sql.Write(string.Join(",", CommonDirectives.Select(x => x.ToSql())));
        }
        return sql.ToString();
    }
}