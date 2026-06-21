using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateResourceGovernorObjectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateResourceGovernorObjectStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateResourceGovernorObjectStatement(this);
    }

    public string Kind { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE {Kind} {Name}");
        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
