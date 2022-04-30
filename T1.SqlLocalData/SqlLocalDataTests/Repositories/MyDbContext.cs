using System.Collections.Generic;
using Dapper;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;

namespace SqlLocalDataTests.Repositories;

public class MyDbContext : DbContext
{
	//string _connectionString = "AttachDBFilename={databaseName}.mdf";
	string _connectionString = "Server=(localdb)\\localtest;Integrated security=SSPI;database=test;";

	public DbSet<CustomerEntity> Customers { get; set; }

	public IEnumerable<T> QueryRawSql<T>(string sql, object parameter = null)
	{
		using var conn = Database.GetDbConnection();
		return conn.Query<T>(sql, parameter);
	}

	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		optionsBuilder.UseSqlServer(_connectionString)
			.UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);
	}
}
