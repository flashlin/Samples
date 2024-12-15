using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForXmlPathClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ForXmlPathClause;
    public string? PathName { get; set; }
    public List<ISqlExpression> CommonDirectives { get; set; } = [];

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
        sql.Write(string.Join(",", CommonDirectives.Select(x => x.ToSql())));
        return sql.ToString();
    }
}