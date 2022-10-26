using System.Data;
using System.Data.Common;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace EFSample;

public static class DbHelper
{
    public static List<T> RawSqlQuery<T>(this DbContext context, string query, Func<DbDataReader, T> map)
    {
        using var command = context.Database.GetDbConnection().CreateCommand();
        command.CommandText = query;
        command.CommandType = CommandType.Text;

        context.Database.OpenConnection();

        using var result = command.ExecuteReader();
        var entities = new List<T>();
        while (result.Read())
        { 
            entities.Add(map(result));
        }
        return entities;
    }
}

public static class DbContextOptionsBuilder
{
    public static DbContextOptions<T> UseSqlServer<T>(string connectionString)
        where T : DbContext
    {
        return new DbContextOptionsBuilder<T>()
            .UseSqlServer(connectionString)
            .Options;
    }

    public static DbContextOptions<T> UseSqliteMemory<T>(string dbname)
        where T : DbContext
    {
        var connection = new SqliteConnection("Filename=:memory:");
        connection.Open();
        return new DbContextOptionsBuilder<T>()
            //.UseSqlite($"Data Source={dbname};Mode=Memory;Cache=Shared")
            .UseSqlite(connection)
            .Options;
    }
}