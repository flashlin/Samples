using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public interface ISqlConstraint : ISqlExpression
{
    string ConstraintName { get; set; }
}