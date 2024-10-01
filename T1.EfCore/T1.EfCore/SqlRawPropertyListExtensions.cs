namespace T1.EfCore;

public static class SqlRawPropertyListExtensions
{
    public static string CreateMemTableSql(this List<SqlRawProperty> dataSqlRawProperties)
    {
        return $"CREATE TABLE #TempMemoryTable ({CreateTableColumnsTypes(dataSqlRawProperties)});";
    }

    private static string CreateTableColumnsTypes(List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"[{x.PropertyName}] {x.DataValue.GetColumnType()}"));
    }
}