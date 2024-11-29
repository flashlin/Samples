using SqlSharpLit.Common.ParserLit.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public interface ISqlValue: ISqlExpression
{
    string Value { get; }
} 
