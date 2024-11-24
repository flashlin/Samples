namespace SqlSharpLit.Common.ParserLit;

public interface ISqlConstraint : ISqlExpression
{
    string ConstraintName { get; set; }
}