using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public interface ISqlValue : ISqlExpression
{
    string Value { get; }
} 
