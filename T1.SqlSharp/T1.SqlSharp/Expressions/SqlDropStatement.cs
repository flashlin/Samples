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
    Type,
    Synonym,
    Login,
    User,
    Role
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
    public string TypeName { get; set; } = string.Empty;
    public bool IfExists { get; set; }
    public List<string> Names { get; set; } = [];
    public string OnTable { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        var typeText = string.IsNullOrEmpty(TypeName) ? ObjectType.ToString().ToUpper() : TypeName;
        sql.Append($"DROP {typeText}");
        if (IfExists)
        {
            sql.Append(" IF EXISTS");
        }
        if (Names.Count > 0)
        {
            sql.Append($" {string.Join(", ", Names)}");
        }
        if (!string.IsNullOrEmpty(OnTable))
        {
            sql.Append($" ON {OnTable}");
        }
        return sql.ToString();
    }
}
