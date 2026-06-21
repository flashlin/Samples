namespace T1.SqlSharp.Expressions;

public class SqlReturnStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ReturnStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ReturnStatement(this);
    }

    public ISqlExpression? Value { get; set; }

    public string ToSql()
    {
        return Value != null ? $"RETURN {Value.ToSql()}" : "RETURN";
    }
}
