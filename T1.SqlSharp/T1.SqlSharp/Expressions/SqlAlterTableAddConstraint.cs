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
    public bool? WithCheck { get; set; }

    public string ToSql()
    {
        var prefix = WithCheck switch
        {
            true => "WITH CHECK ",
            false => "WITH NOCHECK ",
            _ => string.Empty
        };
        return $"{prefix}ADD {Constraint.ToSql()}";
    }
}
