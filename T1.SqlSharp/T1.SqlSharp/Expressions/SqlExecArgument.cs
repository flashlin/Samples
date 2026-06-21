namespace T1.SqlSharp.Expressions;

public class SqlExecArgument : ISqlExpression
{
    public SqlType SqlType => SqlType.ExecArgument;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ExecArgument(this);
    }

    public string ParameterName { get; set; } = string.Empty;
    public required ISqlExpression Value { get; set; }
    public bool IsOutput { get; set; }

    public string ToSql()
    {
        var sql = $"{ParameterName} = {Value.ToSql()}";
        return IsOutput ? $"{sql} OUTPUT" : sql;
    }
}
