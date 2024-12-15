namespace T1.SqlSharp.Expressions;

public interface ISqlExpression
{
    SqlType SqlType { get; }
    string ToSql();
    TextSpan Span { get; set; }
}