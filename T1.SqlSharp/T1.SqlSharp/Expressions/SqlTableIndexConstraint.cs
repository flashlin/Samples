using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlTableIndexConstraint : ISqlConstraint
{
    public SqlType SqlType => SqlType.TableIndexConstraint;
    public TextSpan Span { get; set; } = new();
    public string ConstraintName { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
    public string Clustered { get; set; } = string.Empty;
    public List<SqlConstraintColumn> Columns { get; set; } = [];

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TableIndexConstraint(this);
    }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"INDEX {IndexName}");
        if (!string.IsNullOrEmpty(Clustered))
        {
            sql.Append($" {Clustered}");
        }
        sql.Append($" ({string.Join(", ", Columns.Select(column => column.ToSql()))})");
        return sql.ToString();
    }
}
