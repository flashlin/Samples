using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlInnerTableSource : SqlTableSource
{
    public new SqlType SqlType { get; } = SqlType.InnerTableSource;
    public required ISqlExpression Inner { get; set; }
    
    public override string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Inner.ToSql());
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Write($" AS {Alias}");
        }
        if (Withs.Count > 0)
        {
            sql.Write(" WITH(");
            foreach (var with in Withs.Select((value,index)=> new {value, index}))
            {
                sql.Write($"{with.value.ToSql()}");
                if (with.index < Withs.Count - 1)
                {
                    sql.Write(", ");
                }
            }
            sql.Write(")");
        }
        return TableName;
    }
}