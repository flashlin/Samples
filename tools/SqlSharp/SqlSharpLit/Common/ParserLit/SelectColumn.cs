using System.Text;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SelectColumn : ISelectColumnExpression
{
    public SqlType SqlType { get; } = SqlType.SelectColumn;
    public string ColumnName { get; set; } = string.Empty;
    public string Alias { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"{ColumnName}");
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Append($" AS {Alias}");
        }
        return sql.ToString();
    }
}