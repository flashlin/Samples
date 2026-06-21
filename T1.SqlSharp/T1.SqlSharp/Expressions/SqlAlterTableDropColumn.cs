namespace T1.SqlSharp.Expressions;

public class SqlAlterTableDropColumn : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableDropColumn;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableDropColumn(this);
    }

    public List<string> ColumnNames { get; set; } = [];

    public string ToSql()
    {
        return $"DROP COLUMN {string.Join(", ", ColumnNames)}";
    }
}
