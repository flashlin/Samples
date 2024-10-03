using System.Data;
using System.Data.Common;
using Microsoft.EntityFrameworkCore;

namespace T1.EfCore;

public static class DbContextExtensions
{
    public static void ExecuteDbCommand(this DbConnection connection, string sql)
    {
        using var dbCommand = connection.CreateCommand();
        dbCommand.CommandText = sql;
        dbCommand.ExecuteNonQuery();
    }

    public static DbConnection OpenDbConnection(this DbContext dbContext)
    {
        var connection = dbContext.Database.GetDbConnection();
        if(connection.State != ConnectionState.Open)
            connection.Open();
        return connection;
    }
}