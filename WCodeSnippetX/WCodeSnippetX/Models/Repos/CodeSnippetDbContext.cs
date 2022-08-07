using Microsoft.EntityFrameworkCore;

namespace WCodeSnippetX.Models.Repos;

public class CodeSnippetDbContext : DbContext
{
	public static readonly string DbFilename = "codeSnippet.db";

	public DbSet<CodeSnippetEntity> CodeSnippets { get; set; }

	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		var baseDir = AppDomain.CurrentDomain.BaseDirectory;
		var dbFile = Path.Combine(baseDir, DbFilename);
		optionsBuilder.UseSqlite($"DataSource={dbFile};")
			.UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);
	}

	protected override void OnModelCreating(ModelBuilder modelBuilder)
	{
		modelBuilder.Entity<CodeSnippetEntity>(entity =>
		{
			entity.HasKey(e => new { e.Id });
		});
		//modelBuilder.Entity<TransEntity>()
		//	.Property(e => e.Balance)
		//	.HasConversion<double>();
	}
}