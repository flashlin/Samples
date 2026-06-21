namespace T1.SqlSharp.Expressions;

public class SqlAlterTableAddElements : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableAddElements;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableAddElements(this);
    }

    public List<SqlColumnDefinition> Columns { get; set; } = [];
    public List<ISqlConstraint> Constraints { get; set; } = [];

    public string ToSql()
    {
        var parts = new List<string>();
        parts.AddRange(Columns.Select(column => column.ToSql()));
        parts.AddRange(Constraints.Select(constraint => constraint.ToSql()));
        return $"ADD {string.Join(", ", parts)}";
    }
}
