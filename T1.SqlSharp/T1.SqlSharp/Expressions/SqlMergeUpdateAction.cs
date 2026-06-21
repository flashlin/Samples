namespace T1.SqlSharp.Expressions;

public class SqlMergeUpdateAction : ISqlMergeAction
{
    public SqlType SqlType => SqlType.MergeUpdateAction;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_MergeUpdateAction(this);
    }

    public List<SqlAssignExpr> SetClauses { get; set; } = [];

    public string ToSql()
    {
        return $"UPDATE SET {string.Join(", ", SetClauses.Select(c => c.ToSql()))}";
    }
}
