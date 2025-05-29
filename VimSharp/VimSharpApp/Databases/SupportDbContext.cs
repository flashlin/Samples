using Microsoft.EntityFrameworkCore;

namespace VimSharpApp.Databases
{
    // DbContext for SQLite, no DbSet yet
    public class SupportDbContext : DbContext
    {
        private const string DbFile = "./Support.db";

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!File.Exists(DbFile))
            {
                Database.EnsureCreated();
            }
            // Use SQLite database
            optionsBuilder.UseSqlite($"Data Source={DbFile}");
        }
    }
} 