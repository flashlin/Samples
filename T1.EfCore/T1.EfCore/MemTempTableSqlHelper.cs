using System.Text;

namespace T1.EfCore;

public static class MemTempTableSqlHelper
{
    public static string CreateInsertIntoMemTempTableSql(this List<List<SqlRawProperty>> dataSqlRawPropertiesRows,
        string insertColumns)
    {
        var sql = new StringBuilder();
        foreach (var entityRawProperties in dataSqlRawPropertiesRows)
        {
            var insertRowIntoMemTempTableSql = entityRawProperties.CreateInsertRowIntoMemTempTableSql(insertColumns);
            sql.AppendLine(insertRowIntoMemTempTableSql);
        }

        return sql.ToString();
    }

    public static string CreateInsertRowIntoMemTempTableSql(this List<SqlRawProperty> rawProperties, string insertColumns)
    {
        var insertValues = rawProperties.CreateInsertValuesSql();
        return $@"INSERT INTO #TempMemoryTable ({insertColumns}) VALUES ({insertValues});";
    }

    public static string CreateInsertValuesSql(this List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"@p{x.DataValue.ArgumentIndex}"));
    }

    public static string CreateMemTableSql(this List<SqlRawProperty> dataSqlRawProperties, string tableName)
    {
        return $"CREATE TABLE {tableName} ({CreateTableColumnsTypes(dataSqlRawProperties)});";
    }

    private static string CreateTableColumnsTypes(List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"[{x.PropertyName}] {x.DataValue.GetColumnType()}"));
    }

    public static string CreateAndInsertMemTempTableSql(this List<List<SqlRawProperty>> dataSqlRawPropertiesRows,
        string insertColumns)
    {
        var createMemTableSql = dataSqlRawPropertiesRows[0].CreateMemTableSql("#TempMemoryTable");
        var insertMemTableSql = dataSqlRawPropertiesRows.CreateInsertIntoMemTempTableSql(insertColumns);
        return createMemTableSql + "\n" + insertMemTableSql;
    }
}