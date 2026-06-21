namespace T1.SqlSharp.Expressions;

public class SqlAlterTableSet : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableSet;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableSet(this);
    }

    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        return $"SET ({string.Join(", ", Options)})";
    }
}
