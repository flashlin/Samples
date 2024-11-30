namespace T1.SqlSharp.Expressions;

public class SqlConstraintDefaultValue : ISqlConstraint
{
    public SqlType SqlType { get; } = SqlType.ConstraintDefaultValue;
    public string ConstraintName { get; set; } = string.Empty;
    public string DefaultValue { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"DEFAULT {DefaultValue}";
    }
}