using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateEndpointStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateEndpointStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateEndpointStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;
    public string Protocol { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE ENDPOINT {Name}");
        if (!string.IsNullOrEmpty(State))
        {
            sql.Append($" STATE = {State}");
        }

        if (!string.IsNullOrEmpty(Protocol))
        {
            sql.Append($" AS {Protocol} ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
