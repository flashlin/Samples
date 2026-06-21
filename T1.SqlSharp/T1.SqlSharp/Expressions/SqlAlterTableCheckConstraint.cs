namespace T1.SqlSharp.Expressions;

public class SqlAlterTableCheckConstraint : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableCheckConstraint;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableCheckConstraint(this);
    }

    public bool Check { get; set; }
    public bool AllConstraints { get; set; }
    public List<string> ConstraintNames { get; set; } = [];

    public string ToSql()
    {
        var keyword = Check ? "CHECK" : "NOCHECK";
        var target = AllConstraints ? "ALL" : string.Join(", ", ConstraintNames);
        return $"{keyword} CONSTRAINT {target}";
    }
}
