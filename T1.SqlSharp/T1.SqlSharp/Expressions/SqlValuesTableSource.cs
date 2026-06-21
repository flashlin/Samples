using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlValuesTableSource : SqlTableSource
{
    public new SqlType SqlType { get; } = SqlType.ValuesTableSource;
    public List<List<ISqlExpression>> Rows { get; set; } = [];
    public List<string> ColumnAliases { get; set; } = [];

    public override string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("(VALUES ");
        sql.Append(string.Join(", ", Rows.Select(row => $"({string.Join(", ", row.Select(v => v.ToSql()))})")));
        sql.Append(')');
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Append($" AS {Alias}");
            if (ColumnAliases.Count > 0)
            {
                sql.Append($" ({string.Join(", ", ColumnAliases)})");
            }
        }

        return sql.ToString();
    }
}
