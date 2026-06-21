namespace T1.SqlSharp.Expressions;

public class SqlAlterTableDropConstraint : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableDropConstraint;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableDropConstraint(this);
    }

    public List<string> ConstraintNames { get; set; } = [];

    public string ToSql()
    {
        return $"DROP CONSTRAINT {string.Join(", ", ConstraintNames)}";
    }
}
