using Microsoft.EntityFrameworkCore;
using SqliteCli.Entities;

namespace SqliteCli.Repos
{
	public class StockDatabase : DbContext
	{
		private readonly string _sqliteFile;

		public StockDatabase(string sqliteFile)
		{
			_sqliteFile = sqliteFile;
		}

		public DbSet<StockEntity> StocksMap { get; set; }
		public DbSet<TransEntity> Trans { get; set; }
		public DbSet<StockHistoryEntity> StocksHistory { get; set; }

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			optionsBuilder.UseSqlite($"DataSource={_sqliteFile};");
		}
	}

}
