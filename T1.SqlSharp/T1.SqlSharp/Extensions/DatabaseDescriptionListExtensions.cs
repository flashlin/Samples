using T1.SqlSharp.DatabaseDescriptions;
using T1.Standard.Linq;

namespace T1.SqlSharp.Extensions;

public static class DatabaseDescriptionListExtensions
{
    public static void UpdateDatabaseDescription(this List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        var tmpDatabases = databasesDesc.ToList();
        foreach (var database in tmpDatabases)
        {
            var userDatabase = userDatabaseDesc.FirstOrDefault(x => x.DatabaseName.IsSameAs(database.DatabaseName));
            if (userDatabase == null)
            {
                continue;
            }
            database.Description = string.IsNullOrEmpty(userDatabase.Description) ? database.Description : userDatabase.Description;
            database.Tables.UpdateTableColumnsDescription(userDatabase.Tables);
        }
    }

    public static void UpdateTableColumnsDescription(this List<TableDescription> tables,
        List<TableDescription> userTables)
    {
        var tmpTables = tables.ToList();
        foreach (var table in tmpTables)
        {
            var userTable = userTables.FirstOrDefault(x => x.TableName.IsSameAs(table.TableName));
            if (userTable == null)
            {
                continue;
            }
            table.UpdateColumnsDescription(userTable);
        }
    }
    
    public static void MergeDatabases(this List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> newestDatabaseDesc)
    {
        var tmpDatabases = databasesDesc.ToList();
        foreach (var database in tmpDatabases)
        {
            var newDatabase = newestDatabaseDesc.FirstOrDefault(x => x.DatabaseName.IsSameAs(database.DatabaseName));
            if (newDatabase == null)
            {
                databasesDesc.Remove(database);
                continue;
            }
            database.Tables.MergeTables(newDatabase.Tables);
        }
        foreach (var userDatabase in newestDatabaseDesc)
        {
            var database = databasesDesc.FirstOrDefault(x => x.DatabaseName.IsSameAs(userDatabase.DatabaseName));
            if (database == null)
            {
                databasesDesc.Add(userDatabase);
            }
        }
    }

    public static void MergeTables(this List<TableDescription> tables, List<TableDescription> newestTables)
    {
        var tmpTables = tables.ToList();
        foreach (var table in tmpTables)
        {
            var userTable = newestTables.FirstOrDefault(x => x.TableName.IsSameAs(table.TableName));
            if (userTable == null)
            {
                tables.Remove(table);
                continue;
            }
            table.UpdateColumnsDescription(userTable);
        }
        foreach (var userTable in newestTables)
        {
            var table = tables.FirstOrDefault(x => x.TableName.IsSameAs(userTable.TableName));
            if (table == null)
            {
                tables.Add(userTable);
            }
        }
    }
}