namespace T1.SqlSharp.Expressions;

public class SqlAlterTableAlterColumn : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableAlterColumn;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableAlterColumn(this);
    }

    public required SqlColumnDefinition Column { get; set; }

    public string ToSql()
    {
        return $"ALTER COLUMN {Column.ToSql()}";
    }
}
