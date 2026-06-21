using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlTryCatchStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.TryCatchStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TryCatchStatement(this);
    }

    public List<ISqlExpression> TryStatements { get; set; } = [];
    public List<ISqlExpression> CatchStatements { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("BEGIN TRY ");
        sql.Append(string.Join("; ", TryStatements.Select(s => s.ToSql())));
        sql.Append(" END TRY BEGIN CATCH ");
        sql.Append(string.Join("; ", CatchStatements.Select(s => s.ToSql())));
        sql.Append(" END CATCH");
        return sql.ToString();
    }
}
