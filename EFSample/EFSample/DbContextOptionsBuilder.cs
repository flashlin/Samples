using Microsoft.EntityFrameworkCore;

namespace EFSample;

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
        return new DbContextOptionsBuilder<T>()
            .UseSqlite($"Data Source={dbname};Mode=Memory;Cache=Shared")
            .Options;
    }
}