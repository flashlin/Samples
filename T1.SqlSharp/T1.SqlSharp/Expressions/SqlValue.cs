using T1.Standard.IO;
using YamlDotNet.Core.Tokens;

namespace T1.SqlSharp.Expressions;

public class SqlValue : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.String;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{Value}";
    }
}

public class SqlUnpivotClause : ISqlExpression
{
    public SqlType SqlType => SqlType.UnpivotClause;
    public required ISqlExpression NewColumn { get; set; }
    public required ISqlExpression ForSource { get; set; }
    public List<ISqlExpression> InColumns { get; set; } = [];
    public string AliasName { get; set; } = string.Empty; 

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("UNPIVOT (");
        sql.WriteLine();
        sql.Indent++;
        sql.Write(NewColumn.ToSql());
        sql.WriteLine();
        sql.Write("FOR ");
        sql.Write(ForSource.ToSql());
        sql.WriteLine();
        sql.Write("IN (");
        sql.WriteLine();
        sql.Indent++;
        foreach (var item in InColumns.Select((column, index) => new { column, index }))
        {
            sql.Write(item.column.ToSql());
            if (item.index < InColumns.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        sql.Indent--;
        sql.Write(") ");
        sql.WriteLine($"AS {AliasName}");
        sql.Indent--;
        return sql.ToString();
    }
}