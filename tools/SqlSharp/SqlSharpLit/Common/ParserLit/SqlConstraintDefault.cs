namespace SqlSharpLit.Common.ParserLit;

public class SqlConstraintDefault : ISqlConstraint
{
    public SqlType SqlType => SqlType.ConstraintDefault;
    public string ConstraintName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"CONSTRAINT {ConstraintName} DEFAULT ({Value})";
    }
}