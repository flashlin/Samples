using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlValues : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Values;
    public List<ISqlExpression> Items { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("(");
        for (var i = 0; i < Items.Count; i++)
        {
            sql.Append(Items[i].ToSql());
            if (i < Items.Count - 1)
            {
                sql.Append(", ");
            }
        }
        sql.Append(")");
        return sql.ToString();
    }

}