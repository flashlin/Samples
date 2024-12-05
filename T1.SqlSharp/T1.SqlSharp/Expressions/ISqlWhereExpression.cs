namespace T1.SqlSharp.Expressions;

public interface ISqlWhereExpression : ISqlExpression
{
    string ToSql();
}