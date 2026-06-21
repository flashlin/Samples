using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateTypeStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateTypeStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateTypeStatement(this);
    }

    public string TypeName { get; set; } = string.Empty;
    public bool IsTable { get; set; }
    public List<SqlColumnDefinition> TableColumns { get; set; } = [];
    public string BaseType { get; set; } = string.Empty;
    public SqlDataSize? BaseSize { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE TYPE {TypeName}");
        if (IsTable)
        {
            sql.Append(" AS TABLE (");
            sql.Append(string.Join(", ", TableColumns.Select(c => c.ToSql())));
            sql.Append(')');
        }
        else
        {
            sql.Append($" FROM {BaseType}");
            if (BaseSize != null)
            {
                sql.Append(BaseSize.ToSql());
            }
        }

        return sql.ToString();
    }
}
