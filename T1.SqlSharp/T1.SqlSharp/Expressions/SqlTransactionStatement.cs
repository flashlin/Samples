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
    public bool WithMark { get; set; }
    public string MarkDescription { get; set; } = string.Empty;
    public List<string> WithOptions { get; set; } = [];

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
        var sql = string.IsNullOrEmpty(Name) ? keyword : $"{keyword} {Name}";
        if (WithMark)
        {
            return string.IsNullOrEmpty(MarkDescription) ? $"{sql} WITH MARK" : $"{sql} WITH MARK {MarkDescription}";
        }

        if (WithOptions.Count > 0)
        {
            return $"{sql} WITH ({string.Join(", ", WithOptions)})";
        }

        return sql;
    }
}
