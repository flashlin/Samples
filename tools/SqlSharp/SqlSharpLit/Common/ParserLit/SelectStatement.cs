using SqlSharpLit.Common.ParserLit.Expressions;
using T1.Standard.IO;

namespace SqlSharpLit.Common.ParserLit;

public class SelectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.Select;
    public string AllOrDistinct { get; set; } = string.Empty;
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public ISelectFromExpression From { get; set; } = new SelectFrom();
    public ISqlWhereExpression? Where { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("SELECT");
        if (!string.IsNullOrEmpty(AllOrDistinct))
        {
            sql.Write($" {AllOrDistinct}");
        }
        sql.WriteLine();
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