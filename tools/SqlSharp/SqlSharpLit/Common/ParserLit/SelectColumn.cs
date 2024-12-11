using System.Text;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SelectColumn : ISelectColumnExpression
{
    public SqlType SqlType { get; } = SqlType.SelectColumn;
    public required ISqlExpression Field { get; set; }
    public string Alias { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"({Field.ToSql()})");
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Append($" AS {Alias}");
        }
        return sql.ToString();
    }
}