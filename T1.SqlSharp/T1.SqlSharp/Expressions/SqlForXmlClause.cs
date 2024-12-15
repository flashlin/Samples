using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForXmlClause : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.ForXmlClause;
    public ForXmlType XmlType { get; set; } = ForXmlType.Auto;
    public string Path { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write($"FOR XML {XmlType.ToString().ToUpper()}");
        if (XmlType == ForXmlType.Path)
        {
            sql.Write($"({Path})");
        }
        return sql.ToString();
    }
}