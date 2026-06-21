using System.Text;
using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlFuncTableSource : SqlTableSource
{
    public new SqlType SqlType { get; } = SqlType.FuncTableSource;
    public required SqlFunctionExpression Function { get; set; }
    public List<string> JsonSchemaColumns { get; set; } = [];
    public override string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Function.ToSql());
        if (JsonSchemaColumns.Count > 0)
        {
            sql.Write($" WITH ({string.Join(", ", JsonSchemaColumns)})");
        }

        WriteSqlAfterTableName(sql);
        return sql.ToString();
    }
}