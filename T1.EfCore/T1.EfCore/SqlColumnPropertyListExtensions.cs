using System.Data;

namespace T1.EfCore;

public static class SqlColumnPropertyListExtensions
{
    public static DataTable CreateDataTable(this List<SqlColumnProperty> sqlColumnProperties)
    {
        var dataTable = new DataTable();
        foreach (var column in sqlColumnProperties)
        {
            var dataColumn = new DataColumn(column.ColumnName, column.Property.ClrType); 
            dataTable.Columns.Add(dataColumn);
        }
        return dataTable;
    }

    public static void AddData(this DataTable dataTable, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        foreach (var entity in dataSqlRawProperties)
        {
            var row = dataTable.NewRow();
            foreach (var prop in entity)
            {
                row[prop.ColumnName] = prop.DataValue.Value;
            }
            dataTable.Rows.Add(row);
        }
    }
}