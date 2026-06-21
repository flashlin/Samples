namespace T1.SqlSharp.Expressions;

public enum SqlTransactionAction
{
    Begin,
    Commit,
    Rollback,
    Save
}

public class SqlTransactionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.TransactionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TransactionStatement(this);
    }

    public SqlTransactionAction Action { get; set; }
    public string Name { get; set; } = string.Empty;
    public bool IsDistributed { get; set; }

    public string ToSql()
    {
        var keyword = Action switch
        {
            SqlTransactionAction.Begin => IsDistributed ? "BEGIN DISTRIBUTED TRANSACTION" : "BEGIN TRANSACTION",
            SqlTransactionAction.Commit => "COMMIT TRANSACTION",
            SqlTransactionAction.Rollback => "ROLLBACK TRANSACTION",
            SqlTransactionAction.Save => "SAVE TRANSACTION",
            _ => string.Empty
        };
        return string.IsNullOrEmpty(Name) ? keyword : $"{keyword} {Name}";
    }
}
