using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlUpdateStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.UpdateStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_UpdateStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public List<SqlSetColumn> SetColumns { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"UPDATE {TableName} SET ");
        sql.Append(string.Join(", ", SetColumns.Select(c => $"[{c.ColumnName}] = {c.ParameterName}")));
        return sql.ToString();
    }
}

