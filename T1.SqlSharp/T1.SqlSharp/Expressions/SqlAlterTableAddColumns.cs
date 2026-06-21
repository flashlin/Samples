namespace T1.SqlSharp.Expressions;

public class SqlAlterTableAddColumns : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableAddColumns;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableAddColumns(this);
    }

    public List<SqlColumnDefinition> Columns { get; set; } = [];

    public string ToSql()
    {
        return $"ADD {string.Join(", ", Columns.Select(c => c.ToSql()))}";
    }
}
