using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateAggregateStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateAggregateStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateAggregateStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<string> Parameters { get; set; } = [];
    public string ReturnType { get; set; } = string.Empty;
    public string ExternalName { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE AGGREGATE {Name} ({string.Join(", ", Parameters)}) RETURNS {ReturnType}");
        if (!string.IsNullOrEmpty(ExternalName))
        {
            sql.Append($" EXTERNAL NAME {ExternalName}");
        }

        return sql.ToString();
    }
}
