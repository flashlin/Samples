namespace SqlSharpLit.Common.ParserLit;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
}