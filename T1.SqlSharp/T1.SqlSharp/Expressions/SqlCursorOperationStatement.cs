namespace T1.SqlSharp.Expressions;

public enum SqlCursorOperation
{
    Open,
    Close,
    Deallocate
}

public class SqlCursorOperationStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CursorOperationStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CursorOperationStatement(this);
    }

    public SqlCursorOperation Action { get; set; }
    public string CursorName { get; set; } = string.Empty;
    public bool IsGlobal { get; set; }

    public string ToSql()
    {
        var global = IsGlobal ? "GLOBAL " : string.Empty;
        return $"{Action.ToString().ToUpperInvariant()} {global}{CursorName}";
    }
}
