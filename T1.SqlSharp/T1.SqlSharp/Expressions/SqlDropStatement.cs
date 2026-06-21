using System.Text;

namespace T1.SqlSharp.Expressions;

public enum SqlDropObjectType
{
    Table,
    View,
    Procedure,
    Function,
    Index,
    Trigger,
    Schema,
    Database,
    Sequence,
    Type
}

public class SqlDropStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.DropStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DropStatement(this);
    }

    public SqlDropObjectType ObjectType { get; set; }
    public bool IfExists { get; set; }
    public List<string> Names { get; set; } = [];
    public string OnTable { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"DROP {ObjectType.ToString().ToUpper()}");
        if (IfExists)
        {
            sql.Append(" IF EXISTS");
        }
        sql.Append($" {string.Join(", ", Names)}");
        if (!string.IsNullOrEmpty(OnTable))
        {
            sql.Append($" ON {OnTable}");
        }
        return sql.ToString();
    }
}
