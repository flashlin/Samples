using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlCaseExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Case;
    public ISqlExpression? Input { get; set; }
    public List<SqlCaseWhenClause> Whens { get; set; } = [];
    public ISqlExpression? Else { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("CASE");
        if (Input != null)
        {
            sql.Write(" " + Input.ToSql());
        }
        sql.WriteLine();
        sql.Indent++;
        foreach (var when in Whens)
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