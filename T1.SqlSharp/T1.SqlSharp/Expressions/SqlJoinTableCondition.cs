using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlJoinTableCondition : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.JoinCondition; 
    public JoinType JoinType { get; set; } = JoinType.Inner;
    public required ITableSource JoinedTable { get; set; }
    public required ISqlExpression OnCondition { get; set; }
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(JoinType.ToString().ToUpper());
        sql.Write(" JOIN ");
        sql.Write(JoinedTable.ToSql());
        sql.Write(" ON ");
        sql.Write(OnCondition.ToSql());
        return sql.ToString();
    }
}