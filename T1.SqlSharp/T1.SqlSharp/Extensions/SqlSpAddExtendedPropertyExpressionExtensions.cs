using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public static class SqlSpAddExtendedPropertyExpressionExtensions
{
    public static List<SqlSpAddExtendedPropertyExpression> FilterByTableName(
        this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string tableName)
    {
        return spAddExtendedPropertyExpressions.Where(x=>
            x.Level1Type.IsNormalizeSameAs("TABLE") && x.Level1Name.IsNormalizeSameAs(tableName))
            .ToList();
    }

    public static List<SqlSpAddExtendedPropertyExpression> FilterByColumnName(
        this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string columnName)
    {
        return spAddExtendedPropertyExpressions
            .Where(x => x.Level2Type.IsNormalizeSameAs("COLUMN") && x.Level2Name.IsNormalizeSameAs(columnName))
            .ToList();
    }
    
    public static string GetColumnDescription(this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string tableName, string columnName)
    {
        return spAddExtendedPropertyExpressions
            .FilterByTableName(tableName)
            .FilterByColumnName(columnName)
            .Where(x => x.Name.IsNormalizeSameAs("MS_Description"))
            .Select(x => x.Value.NormalizeName())
            .FirstOrDefault(string.Empty);
    }
}