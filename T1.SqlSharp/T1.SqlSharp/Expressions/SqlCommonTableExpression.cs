using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlCommonTableExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.CommonTableExpression;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CommonTableExpression(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<string> ColumnNames { get; set; } = [];
    public required ISqlExpression Query { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Name);
        if (ColumnNames.Count > 0)
        {
            sql.Write($" ({string.Join(", ", ColumnNames)})");
        }
        sql.WriteLine(" AS (");
        sql.Indent++;
        sql.Write(Query.ToSql());
        sql.Indent--;
        sql.Write(")");
        return sql.ToString();
    }
}
