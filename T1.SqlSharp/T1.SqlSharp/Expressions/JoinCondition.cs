using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class JoinCondition : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.JoinCondition; 
    public JoinType JoinType { get; set; }

    public required SqlTableSource JoinedTable { get; set; }

    public string OnCondition { get; set; } = string.Empty; 
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(JoinType.ToString().ToUpper());
        sql.Write(" JOIN ");
        sql.Write(JoinedTable.ToSql());
        if (!string.IsNullOrEmpty(OnCondition))
        {
            sql.Write(" ON ");
            sql.Write(OnCondition);
        }
        return sql.ToString();
    }
}