using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateAssemblyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateAssemblyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateAssemblyStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string From { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE ASSEMBLY {Name}");
        if (!string.IsNullOrEmpty(From))
        {
            sql.Append($" FROM {From}");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
