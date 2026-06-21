namespace T1.SqlSharp.Expressions;

public enum MergeMatchType
{
    Matched,
    NotMatchedByTarget,
    NotMatchedBySource
}

public class SqlMergeWhenClause : ISqlExpression
{
    public SqlType SqlType => SqlType.MergeWhenClause;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_MergeWhenClause(this);
    }

    public MergeMatchType MatchType { get; set; }
    public ISqlExpression? AndCondition { get; set; }
    public required ISqlMergeAction Action { get; set; }

    public string ToSql()
    {
        var matchText = MatchType switch
        {
            MergeMatchType.Matched => "WHEN MATCHED",
            MergeMatchType.NotMatchedByTarget => "WHEN NOT MATCHED BY TARGET",
            MergeMatchType.NotMatchedBySource => "WHEN NOT MATCHED BY SOURCE",
            _ => "WHEN MATCHED"
        };
        var andText = AndCondition != null ? $" AND {AndCondition.ToSql()}" : string.Empty;
        return $"{matchText}{andText} THEN {Action.ToSql()}";
    }
}
