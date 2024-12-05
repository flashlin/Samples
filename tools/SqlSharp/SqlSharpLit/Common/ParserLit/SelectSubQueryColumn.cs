using System.Text;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SelectSubQueryColumn : ISelectColumnExpression
{
    public SelectItemType ItemType => SelectItemType.SubQuery;
    public required ISqlExpression SubQuery { get; set; }
    public string Alias { get; set; } = string.Empty;
    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"({SubQuery.ToSql()})");
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Append($" AS {Alias}");
        }
        return sql.ToString();
    }
}