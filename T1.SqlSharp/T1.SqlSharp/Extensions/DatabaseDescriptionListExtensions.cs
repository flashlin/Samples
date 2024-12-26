using T1.SqlSharp.DatabaseDescriptions;
using T1.Standard.Linq;

namespace T1.SqlSharp.Extensions;

public static class DatabaseDescriptionListExtensions
{
    public static void UpdateDatabaseDescription(this List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        foreach (var database in databasesDesc)
        {
            var userDatabase = userDatabaseDesc.FirstOrDefault(x => x.DatabaseName.IsSameAs(database.DatabaseName)) ??
                               new DatabaseDescription()
                               {
                                   DatabaseName = database.DatabaseName,
                               };
            database.Description = string.IsNullOrEmpty(userDatabase.Description) ? database.Description : userDatabase.Description;
            database.Tables.UpdateTableColumnsDescription(userDatabase.Tables);
        }
    }

    public static void UpdateTableColumnsDescription(this List<TableDescription> tables,
        List<TableDescription> userTables)
    {
        foreach (var table in tables)
        {
            var userTable = userTables.FirstOrDefault(x => x.TableName.IsSameAs(table.TableName)) ??
                            new TableDescription()
                            {
                                TableName = table.TableName,
                            };
            table.UpdateColumnsDescription(userTable);
        }
    }
}