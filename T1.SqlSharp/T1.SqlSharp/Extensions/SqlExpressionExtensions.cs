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

    public static List<SqlCreateTableExpression> FilterCreateTableExpression(this List<ISqlExpression> sqlExpressions)
    {
        return sqlExpressions
            .Where(x => x.SqlType == SqlType.CreateTable)
            .Cast<SqlCreateTableExpression>()
            .Where(x => StartsWithValidChar(x.TableName))
            .ToList();
    }

    private static bool StartsWithValidChar(string text)
    {
        return !string.IsNullOrEmpty(text) && (char.IsLetter(text[0]) || text[0] == '_' || text[0] == '[');
    }
}