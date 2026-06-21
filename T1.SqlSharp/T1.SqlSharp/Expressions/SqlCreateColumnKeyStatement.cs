using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateColumnKeyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateColumnKeyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateColumnKeyStatement(this);
    }

    public bool IsMasterKey { get; set; }
    public string Name { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(IsMasterKey ? "CREATE COLUMN MASTER KEY " : "CREATE COLUMN ENCRYPTION KEY ");
        sql.Append(Name);
        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
