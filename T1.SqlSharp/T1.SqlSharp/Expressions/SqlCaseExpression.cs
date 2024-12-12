using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlCaseExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.CaseExpr;
    public ISqlExpression? When { get; set; }
    public List<SqlWhenThenClause> WhenThens { get; set; } = [];
    public ISqlExpression? Else { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("CASE");
        if (When != null)
        {
            sql.Write(" " + When.ToSql());
        }
        sql.WriteLine();
        sql.Indent++;
        foreach (var when in WhenThens)
        {
            sql.WriteLine(when.ToSql());
        }
        sql.Indent--;
        if (Else != null)
        {
            sql.WriteLine("ELSE " + Else.ToSql());
        }
        sql.WriteLine("END");
        return sql.ToString();
    }
}