namespace T1.SqlSharp.Expressions;

public class SqlTableHintIndex : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.TableHintIndex;
    public TextSpan Span { get; set; } = new();
    public List<ISqlExpression> IndexValues { get; set; } = [];
    public string ToSql()
    {
        return $"INDEX ({string.Join(", ", IndexValues.Select(x => x.ToSql()))})";
    }
}