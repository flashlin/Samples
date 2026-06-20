using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlJoinTableCondition : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.JoinCondition; 
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_JoinTableCondition(this); 
    }

    public JoinType JoinType { get; set; } = JoinType.Inner;
    public required ITableSource JoinedTable { get; set; }
    public ISqlExpression? OnCondition { get; set; }
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(GetJoinKeyword());
        sql.Write(" ");
        sql.Write(JoinedTable.ToSql());
        if (OnCondition != null)
        {
            sql.Write(" ON ");
            sql.Write(OnCondition.ToSql());
        }
        return sql.ToString();
    }

    private string GetJoinKeyword()
    {
        return JoinType switch
        {
            JoinType.Cross => "CROSS JOIN",
            JoinType.CrossApply => "CROSS APPLY",
            JoinType.OuterApply => "OUTER APPLY",
            _ => $"{JoinType.ToString().ToUpper()} JOIN"
        };
    }
}