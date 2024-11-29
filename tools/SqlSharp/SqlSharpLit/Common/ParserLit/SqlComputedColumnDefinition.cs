using System.Text;
using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlComputedColumnDefinition : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ComputedColumn;
    public string ColumnName { get; set; } = string.Empty;
    public string Expression { get; set; } = string.Empty;
    public bool IsPersisted { get; set; } = false;
    public bool IsNotNull { get; set; } = false;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"{ColumnName} AS {Expression}");
        if (IsPersisted)
        {
            sql.Append(" PERSISTED");
        }

        if (IsNotNull)
        {
            sql.Append(" NOT NULL");
        }

        return sql.ToString();
    }
}