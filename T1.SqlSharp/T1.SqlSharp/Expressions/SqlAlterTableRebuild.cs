namespace T1.SqlSharp.Expressions;

public class SqlAlterTableRebuild : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableRebuild;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableRebuild(this);
    }

    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        return Options.Count > 0 ? $"REBUILD WITH ({string.Join(", ", Options)})" : "REBUILD";
    }
}
