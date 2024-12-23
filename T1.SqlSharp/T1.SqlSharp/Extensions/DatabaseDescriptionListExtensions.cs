using T1.SqlSharp.DatabaseDescriptions;
using T1.Standard.Linq;

namespace T1.SqlSharp.Extensions;

public static class DatabaseDescriptionListExtensions
{
    public static List<DatabaseDescription> MergeDatabaseDescription(this List<DatabaseDescription> databasesDesc,
        List<DatabaseDescription> userDatabaseDesc)
    {
        var result = databasesDesc.LeftOuterJoin(userDatabaseDesc,
                udb => udb.DatabaseName,
                db => db.DatabaseName,
                (udb) => udb,
                (db, udb) => new DatabaseDescription()
                {
                    DatabaseName = db.DatabaseName,
                    Description = udb.Description,
                    Tables = db.Tables
                })
            .ToList();
        return result;
    }

    public static void MergeTableColumnsDescription(this List<TableDescription> tables, List<TableDescription> userTables)
    {
        foreach (var table in tables)
        {
            var userTable = userTables.FirstOrDefault(x => x.TableName.IsSameAs(table.TableName)) ?? new TableDescription()
            {
                TableName = table.TableName,
            };
            table.MergeColumnsDescription(userTable);
        }
    }
}