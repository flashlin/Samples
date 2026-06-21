using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlInsertStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.InsertStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_InsertStatement(this);
    }

    public SqlTopClause? Top { get; set; }
    public string TableName { get; set; } = string.Empty;
    public List<ISqlExpression> Withs { get; set; } = [];
    public List<string> Columns { get; set; } = [];
    public List<List<ISqlExpression>> ValuesRows { get; set; } = [];
    public SelectStatement? SourceSelect { get; set; }
    public SqlExecStatement? ExecSource { get; set; }
    public bool IsDefaultValues { get; set; }
    public SqlOutputClause? Output { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("INSERT");
        if (Top != null)
        {
            sql.Append($" {Top.ToSql()}");
        }

        sql.Append($" INTO {TableName}");
        if (Withs.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Withs.Select(hint => hint.ToSql()))})");
        }

        if (Columns.Count > 0)
        {
            sql.Append($" ({string.Join(", ", Columns.Select(column => $"[{column}]"))})");
        }

        if (Output != null)
        {
            sql.Append($" {Output.ToSql()}");
        }

        sql.Append(RenderBody());
        return sql.ToString();
    }

    private string RenderBody()
    {
        if (IsDefaultValues)
        {
            return " DEFAULT VALUES";
        }

        if (SourceSelect != null)
        {
            return $" {SourceSelect.ToSql()}";
        }

        if (ExecSource != null)
        {
            return $" {ExecSource.ToSql()}";
        }

        var rows = ValuesRows.Select(row => $"({string.Join(", ", row.Select(value => value.ToSql()))})");
        return $" VALUES {string.Join(", ", rows)}";
    }
}

