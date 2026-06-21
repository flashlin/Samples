namespace T1.SqlSharp.Expressions;

public class SqlCreateServiceStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateServiceStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateServiceStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string OnQueue { get; set; } = string.Empty;
    public List<string> Contracts { get; set; } = [];

    public string ToSql()
    {
        var contracts = Contracts.Count > 0 ? $" ({string.Join(", ", Contracts)})" : string.Empty;
        return $"CREATE SERVICE {Name} ON QUEUE {OnQueue}{contracts}";
    }
}
