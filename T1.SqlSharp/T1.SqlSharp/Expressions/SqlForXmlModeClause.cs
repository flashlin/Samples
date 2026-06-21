using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForXmlModeClause : ISqlForXmlClause
{
    public SqlType SqlType { get; } = SqlType.ForXmlModeClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ForXmlModeClause(this);
    }

    public SqlForXmlMode Mode { get; set; } = SqlForXmlMode.Raw;
    public string? ElementName { get; set; }
    public List<SqlForXmlRootDirective> CommonDirectives { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write($"FOR XML {GetModeKeyword()}");
        if (ElementName != null)
        {
            sql.Write($"({ElementName})");
        }

        if (CommonDirectives.Count > 0)
        {
            sql.Write(",");
            sql.Write(string.Join(",", CommonDirectives.Select(x => x.ToSql())));
        }

        return sql.ToString();
    }

    private string GetModeKeyword()
    {
        return Mode switch
        {
            SqlForXmlMode.Explicit => "EXPLICIT",
            _ => "RAW"
        };
    }
}
