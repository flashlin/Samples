namespace T1.SqlSharp.Expressions;

public class SqlAlterTableAddConstraint : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableAddConstraint;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableAddConstraint(this);
    }

    public required ISqlConstraint Constraint { get; set; }

    public string ToSql()
    {
        return $"ADD {Constraint.ToSql()}";
    }
}
