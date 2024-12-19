using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlFunctionExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Function;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_FunctionExpression(this);
    }

    public string FunctionName { get; set; } =string.Empty;
    public List<ISqlExpression> Parameters { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(FunctionName);
        sql.Write("(");
        for (var i = 0; i < Parameters.Count; i++)
        {
            sql.Write(Parameters[i].ToSql());
            if (i < Parameters.Count - 1)
            {
                sql.Write(", ");
            }
        }
        sql.Write(")");
        return sql.ToString();
    }

}