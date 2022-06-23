using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GitCli.Models.Repositories
{
	public class GitCliDbContext : DbContext
	{
		private readonly string _sqliteFile = "GitCli.db";

		public DbSet<GitRepositoryEntity> GitRepositories => Set<GitRepositoryEntity>();

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			optionsBuilder.UseSqlite($"DataSource={_sqliteFile};")
				.UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);
		}

		protected override void OnModelCreating(ModelBuilder modelBuilder)
		{
			//modelBuilder.Entity<GitRepositoryEntity>(entity =>
			//{
			//	entity.HasKey(e => new { e.TranDate, e.StockId });
			//});
			//modelBuilder.Entity<GitRepositoryEntity>()
			//	.Property(e => e.Balance)
			//	.HasConversion<double>();
		}
	}

	[Table("GitRepository")]
	public class GitRepositoryEntity
	{
		[Key]
		public string Path { get; set; }
	}
}
