using Microsoft.EntityFrameworkCore;

namespace VimSharpApp.Databases
{
    // DbContext for SQLite, no DbSet yet
    public class SupportDbContext : DbContext
    {
        private readonly string _dbFile;

        public SupportDbContext(string dbFile)
        {
            _dbFile = dbFile;
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            // Use SQLite database
            optionsBuilder.UseSqlite($"Data Source={_dbFile}");
        }
    }
} 