namespace T1.SqlSharp.Expressions;

public class SqlPrintStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.PrintStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_PrintStatement(this);
    }

    public required ISqlExpression Value { get; set; }

    public string ToSql()
    {
        return $"PRINT {Value.ToSql()}";
    }
}
