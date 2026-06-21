using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateExternalStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateExternalStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateExternalStatement(this);
    }

    public string Kind { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public List<SqlColumnDefinition> Columns { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE EXTERNAL {Kind} {Name}");
        if (Columns.Count > 0)
        {
            sql.Append($" ({string.Join(", ", Columns.Select(column => column.ToSql()))})");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
