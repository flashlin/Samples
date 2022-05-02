using System.Collections.Generic;
using System.IO;
using Dapper;
using Microsoft.EntityFrameworkCore;

namespace SqlLocalDbTests.Repositories;

public class MyDbContext : DbContext
{
	string _connectionString = "Server=(localdb)\\localtest;Integrated security=SSPI;database=test;";

	public MyDbContext()
	{		
	}

	public MyDbContext(string connectionString)
	{
		_connectionString = connectionString;
	}

	public DbSet<CustomerEntity> Customers { get; set; }

	public IEnumerable<T> QuerySqlRaw<T>(string sql, object parameter = null)
	{
		using var conn = Database.GetDbConnection();
		return conn.Query<T>(sql, parameter);
	}

	public void ExecuteSqlRaw(string sql, object parameter = null)
	{
		Database.ExecuteSqlRaw(sql, parameter);
	}

	public void ExecuteSqlRawFromFile(string filePath)
	{
		var sql = File.ReadAllText(filePath);
		Database.ExecuteSqlRaw(sql);
	}

	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		optionsBuilder.UseSqlServer(_connectionString)
			.UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);
	}
}
