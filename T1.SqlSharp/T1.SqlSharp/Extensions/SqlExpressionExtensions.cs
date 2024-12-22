using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public static class SqlExpressionExtensions
{
    public static List<SqlSpAddExtendedPropertyExpression> FilterAddExtendedPropertyExpression(
        this List<ISqlExpression> sqlExpressions)
    {
        return sqlExpressions
            .Where(x => x.SqlType == SqlType.AddExtendedProperty)
            .Cast<SqlSpAddExtendedPropertyExpression>()
            .ToList();
    }
}