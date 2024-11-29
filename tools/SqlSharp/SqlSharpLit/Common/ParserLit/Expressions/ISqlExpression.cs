namespace SqlSharpLit.Common.ParserLit.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}