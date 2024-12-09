namespace T1.SqlSharp.Expressions;

public class SqlGroup : ISqlExpression
{
    public SqlType SqlType => SqlType.Group;
    public required ISqlExpression Inner { get; set; }
    public string ToSql()
    {
        return $"({Inner.ToSql()})";
    }
}