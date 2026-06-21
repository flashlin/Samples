using System.Text;

namespace T1.SqlSharp.Expressions;

public enum SqlTriggerTiming
{
    For,
    After,
    InsteadOf
}

public enum SqlTriggerEvent
{
    Insert,
    Update,
    Delete
}

public class SqlCreateTriggerStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateTriggerStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateTriggerStatement(this);
    }

    public bool IsOrAlter { get; set; }
    public bool IsAlter { get; set; }
    public string TriggerName { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public SqlTriggerTiming Timing { get; set; }
    public List<SqlTriggerEvent> Events { get; set; } = [];
    public required ISqlExpression Body { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(DefinitionLead.ToSql(IsAlter, IsOrAlter, "TRIGGER"));
        sql.Append($"{TriggerName} ON {TableName} ");
        sql.Append(TimingToSql());
        sql.Append(' ');
        sql.Append(string.Join(", ", Events.Select(e => e.ToString().ToUpperInvariant())));
        sql.Append($" AS {Body.ToSql()}");
        return sql.ToString();
    }

    private string TimingToSql()
    {
        return Timing switch
        {
            SqlTriggerTiming.For => "FOR",
            SqlTriggerTiming.After => "AFTER",
            SqlTriggerTiming.InsteadOf => "INSTEAD OF",
            _ => string.Empty
        };
    }
}
