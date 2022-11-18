using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;

namespace LinqJoinTutor;

public class MyDbContext : DbContext
{
	public MyDbContext()
		: base(DbOptions())
	{
	}

	public DbSet<Customer> Customers { get; set; }
	public DbSet<House> Houses { get; set; }

	public static DbContextOptions<MyDbContext> DbOptions()
	{
		var sqlBuilder = new SqlConnectionStringBuilder
		{
			DataSource = "localhost\\SQLEXPRESS",
			InitialCatalog = "North",
			IntegratedSecurity = true
		};
		var dbContextOptionsBuilder = new DbContextOptionsBuilder<MyDbContext>();
		return dbContextOptionsBuilder.UseSqlServer(sqlBuilder.ToString())
			.Options;
	}
}