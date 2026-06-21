namespace T1.SqlSharp.Expressions;

public class SqlAlterIndexStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterIndexStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterIndexStatement(this);
    }

    public string IndexName { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public string Partition { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var partition = string.IsNullOrEmpty(Partition) ? string.Empty : $" PARTITION = {Partition}";
        var optionsPrefix = string.Equals(Action, "SET", StringComparison.OrdinalIgnoreCase) ? string.Empty : "WITH ";
        var options = Options.Count > 0 ? $" {optionsPrefix}({string.Join(", ", Options)})" : string.Empty;
        return $"ALTER INDEX {IndexName} ON {TableName} {Action}{partition}{options}";
    }
}
