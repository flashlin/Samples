using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlTableSource : ISqlTableSource
{
    public string TableName { get; set; } = string.Empty;
    public string Alias { get; set; } = string.Empty;
    public JoinCondition? Join { get; set; }
    public ISqlExpression[] Withs { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(TableName);
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Write($" AS {Alias}");
        }
        if (Withs.Length > 0)
        {
            sql.Write(" WITH(");
            foreach (var with in Withs.Select((value,index)=> new {value, index}))
            {
                sql.Write($"{with.value.ToSql()}");
                if (with.index < Withs.Length - 1)
                {
                    sql.Write(", ");
                }
            }
            sql.Write(")");
        }

        if (Join != null)
        {
            sql.Write($" {Join.ToSql()}");
        }

        return TableName;
    }
}