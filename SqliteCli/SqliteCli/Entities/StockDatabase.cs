using Microsoft.EntityFrameworkCore;

namespace SqliteCli.Entities
{
	public class StockDatabase : DbContext
	{
		private readonly string _sqliteFile;

		public StockDatabase(string sqliteFile)
		{
			this._sqliteFile = sqliteFile;
		}

		public DbSet<StockEntity> StocksMap { get; set; }
		public DbSet<TransEntity> Trans { get; set; }

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			optionsBuilder.UseSqlite($"DataSource={_sqliteFile};");
		}
	}

}
