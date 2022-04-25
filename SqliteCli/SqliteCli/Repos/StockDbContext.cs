using Microsoft.EntityFrameworkCore;
using SqliteCli.Entities;

namespace SqliteCli.Repos
{
	public class StockDbContext : DbContext
	{
		private readonly string _sqliteFile = "d:/VDisk/SNL/flash_stock.db";

		public DbSet<StockEntity> StocksMap { get; set; }
		public DbSet<TransEntity> Trans { get; set; }
		public DbSet<StockHistoryEntity> StocksHistory { get; set; }

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			optionsBuilder.UseSqlite($"DataSource={_sqliteFile};");
		}

		protected override void OnModelCreating(ModelBuilder modelBuilder)
		{
			modelBuilder.Entity<StockHistoryEntity>(entity =>
			{
				entity.HasKey(e => new { e.TranDate, e.StockId  });
			});
			modelBuilder.Entity<TransEntity>()
				.Property(e => e.Balance)
				.HasConversion<double>();
		}
	}

}
