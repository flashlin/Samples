using SqlSharpLit.Common.ParserLit.Expressions;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public interface ISqlConstraint : ISqlExpression
{
    string ConstraintName { get; set; }
}