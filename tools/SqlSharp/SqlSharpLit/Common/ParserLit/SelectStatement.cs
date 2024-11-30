using SqlSharpLit.Common.ParserLit.Expressions;
using T1.SqlSharp.Expressions;
using T1.Standard.IO;

namespace SqlSharpLit.Common.ParserLit;

public enum SelectType
{
    All,
    Distinct
}

public enum SelectItemType
{
    /// <summary>
    /// Simple column or expression
    /// </summary>
    Column,

    /// <summary>
    /// Nested subquery
    /// </summary>
    SubQuery
}

public class SelectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.Select;
    public SelectType SelectType { get; set; } = SelectType.All; 
    public SqlTopClause? Top { get; set; }
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public ISqlTableSource From { get; set; } = new SqlTableSource();
    public ISqlWhereExpression? Where { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("SELECT");
        sql.Write($" {SelectType.ToString().ToUpper()}");
        sql.WriteLine();
        if(Top!=null)
        {
            sql.Write(Top.ToSql());
            sql.WriteLine();
        }
        sql.Indent++;
        for (var i = 0; i < Columns.Count; i++)
        {
            sql.Write(Columns[i].ToSql());
            if (i < Columns.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        sql.Indent--;
        sql.WriteLine("FROM");
        sql.Indent++;
        sql.Write(From.ToSql());
        sql.Indent--;
        sql.Indent++;
        if(Where!=null)
        {
            sql.WriteLine("WHERE ");
            sql.Write(Where.ToSql());
        }
        sql.Indent--;
        return sql.ToString();
    }
}