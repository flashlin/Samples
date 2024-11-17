namespace SqlSharpLit.Common.ParserLit;

public class SqlConstraintDefault : ISqlConstraint
{
    public string ConstraintName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
}