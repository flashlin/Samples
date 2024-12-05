using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

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