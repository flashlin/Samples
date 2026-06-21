using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlConstraintCheck : ISqlConstraint
{
    public SqlType SqlType { get; } = SqlType.ConstraintCheck;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ConstraintCheck(this);
    }

    public string ConstraintName { get; set; } = string.Empty;
    public required ISqlExpression Predicate { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        if (!string.IsNullOrEmpty(ConstraintName))
        {
            sql.Write($"CONSTRAINT {ConstraintName} ");
        }
        sql.Write($"CHECK ({Predicate.ToSql()})");
        return sql.ToString();
    }
}
