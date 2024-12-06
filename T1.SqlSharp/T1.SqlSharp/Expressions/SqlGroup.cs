namespace T1.SqlSharp.Expressions;

public class SqlGroup : ISqlValue
{
    public SqlType SqlType => SqlType.Group;
    public required ISqlExpression Inner { get; set; }
    public string Value => Inner.ToSql();
    public string ToSql()
    {
        return $"({Inner.ToSql()})";
    }
}