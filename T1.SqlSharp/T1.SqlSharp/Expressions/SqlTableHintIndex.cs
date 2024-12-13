namespace T1.SqlSharp.Expressions;

public class SqlTableHintIndex : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.TableHintIndex;
    public List<string> IndexValues { get; set; } = [];
    public string ToSql()
    {
        return $"INDEX ({string.Join(", ", IndexValues)})";
    }
}