using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlAlterSequenceStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterSequenceStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterSequenceStatement(this);
    }

    public string SequenceName { get; set; } = string.Empty;
    public bool IsRestart { get; set; }
    public ISqlExpression? RestartWith { get; set; }
    public ISqlExpression? IncrementBy { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"ALTER SEQUENCE {SequenceName}");
        if (IsRestart)
        {
            sql.Append(" RESTART");
            if (RestartWith != null)
            {
                sql.Append($" WITH {RestartWith.ToSql()}");
            }
        }

        if (IncrementBy != null)
        {
            sql.Append($" INCREMENT BY {IncrementBy.ToSql()}");
        }

        return sql.ToString();
    }
}
