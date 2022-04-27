using Microsoft.EntityFrameworkCore;

namespace SqlLocalDataTests.Repositories;

public class MyDbContext : DbContext
{
	//string _connectionString = "AttachDBFilename={databaseName}.mdf";
	string _connectionString = "Server=(localdb)\\localtest;Integrated security=SSPI;database=test;";

	public DbSet<CustomerEntity> Customers { get; set; }

	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		optionsBuilder.UseSqlServer(_connectionString);
	}
}
