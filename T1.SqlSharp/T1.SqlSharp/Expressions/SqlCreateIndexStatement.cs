using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateIndexStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateIndexStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateIndexStatement(this);
    }

    public bool IsUnique { get; set; }
    public string Clustered { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public List<SqlConstraintColumn> Columns { get; set; } = [];
    public List<string> IncludeColumns { get; set; } = [];
    public ISqlExpression? Where { get; set; }
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("CREATE ");
        if (IsUnique)
        {
            sql.Append("UNIQUE ");
        }
        if (!string.IsNullOrEmpty(Clustered))
        {
            sql.Append($"{Clustered} ");
        }
        sql.Append($"INDEX {IndexName} ON {TableName} ");
        sql.Append($"({string.Join(", ", Columns.Select(c => c.ToSql()))})");
        if (IncludeColumns.Count > 0)
        {
            sql.Append($" INCLUDE ({string.Join(", ", IncludeColumns)})");
        }
        if (Where != null)
        {
            sql.Append($" WHERE {Where.ToSql()}");
        }
        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }
        return sql.ToString();
    }
}
