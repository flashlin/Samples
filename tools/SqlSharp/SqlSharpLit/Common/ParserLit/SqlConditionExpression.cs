using T1.SqlSharp.Expressions;
using T1.Standard.IO;

namespace SqlSharpLit.Common.ParserLit;

public enum JoinType
{
    Inner,
    Left,
    Right,
    Full
}

public enum GroupingType
{
    Simple,
    GroupingSets,
    Cube,
    Rollup
}

public enum LogicalOperator
{
    None,
    And,
    Or,
    Not
}

public class TableSource : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.TableSource;
    public string TableName { get; set; } = string.Empty;

    public string Alias { get; set; } = string.Empty;

    public JoinCondition? Join { get; set; }
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(TableName);
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Write($" AS {Alias}");
        }
        if (Join != null)
        {
            sql.Write($" {Join.ToSql()}");
        }
        return sql.ToString();
    }
}

public class JoinCondition : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.JoinCondition; 
    public JoinType JoinType { get; set; }

    public TableSource JoinedTable { get; set; }

    public string OnCondition { get; set; }
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