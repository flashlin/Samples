using System.Text;
using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SelectColumn : ISelectColumnExpression
{
    public SelectItemType ItemType => SelectItemType.Column;
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