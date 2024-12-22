using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public static class SqlSpAddExtendedPropertyExpressionExtensions
{
    public static List<SqlSpAddExtendedPropertyExpression> FilterByTableName(
        this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string tableName)
    {
        return spAddExtendedPropertyExpressions
            .Where(x => x.Name.Contains("MS_Description") && x.Level1Name.IsSameAs(tableName))
            .ToList();
    }

    public static List<SqlSpAddExtendedPropertyExpression> FilterByColumnName(
        this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string columnName)
    {
        return spAddExtendedPropertyExpressions
            .Where(x => x.Name.Contains("MS_Description") && x.Level2Name.IsSameAs(columnName))
            .ToList();
    }
    
    public static string FilterColumnDescription(this List<SqlSpAddExtendedPropertyExpression> spAddExtendedPropertyExpressions, string tableName, string columnName)
    {
        return spAddExtendedPropertyExpressions
            .FilterByTableName(tableName)
            .FilterByColumnName(columnName)
            .Select(x => x.Value)
            .FirstOrDefault(string.Empty);
    }
}