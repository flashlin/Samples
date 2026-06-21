namespace T1.SqlSharp.Expressions;

public class SqlCreateContractStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateContractStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateContractStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<string> Messages { get; set; } = [];

    public string ToSql()
    {
        return $"CREATE CONTRACT {Name} ({string.Join(", ", Messages)})";
    }
}
