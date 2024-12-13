using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SelectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.Select;
    public SelectType SelectType { get; set; } = SelectType.All; 
    public SqlTopClause? Top { get; set; }
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public ISqlExpression From { get; set; } = new SqlTableSource();
    public ISqlExpression? Where { get; set; }
    public SqlOrderByClause? OrderBy { get; set; }

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
        sql.WriteLine("FROM ");
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
        if(OrderBy!=null)
        {
            sql.WriteLine(OrderBy.ToSql());
        }
        return sql.ToString();
    }
}