namespace T1.SqlSharp.Expressions;

public interface ISqlValue : ISqlExpression
{
    string Value { get; }
} 
