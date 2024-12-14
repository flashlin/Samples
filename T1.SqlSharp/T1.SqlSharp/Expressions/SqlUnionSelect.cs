using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlUnionSelect : ISqlExpression
{
    public SqlType SqlType => SqlType.UnionSelect;
    public bool IsAll { get; set; } = false;
    public required SqlSelectStatement SqlSelectStatement { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("UNION ");
        if (IsAll)
        {
            sql.Write("ALL ");
        }
        sql.Write(SqlSelectStatement.ToSql());
        return sql.ToString();
    }
}