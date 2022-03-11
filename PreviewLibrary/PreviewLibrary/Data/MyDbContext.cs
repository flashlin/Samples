using Dapper;
using Microsoft.Data.SqlClient.Server;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using System;
using System.Collections.Generic;
using System.Data;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PreviewLibrary.Data
{

	public class MyDbContext : DbContext
	{
		static readonly string _sampleConnectionString = @"server=.\SQLExpress;Integrated Security=SSPI;database=North;Encrypt=True;TrustServerCertificate=True;";
		//static readonly string _sampleConnectionString = @"server=.\SQLEXPRESS;Persist Security Info=False;User ID=your_id;Password=your_password;database=North;Encrypt=True;TrustServerCertificate=True;";

		public Task<IEnumerable<UserEntity>> GetOrganizationAsync()
		{
			var query = new QueryCommand()
			{
				Command = @"select top 10 Id,Name from [User]",
				Parameters = new List<SqlDataRecord>()
			};

			return QueryAsync<UserEntity>(query);
		}

		public Task<IEnumerable<T>> QueryAsync<T>(QueryCommand queryCommand)
		{
			return QueryBySpAsync<T>("[dbo].[Run_Query]", queryCommand, default);
		}

		private Task<IEnumerable<T>> QueryBySpAsync<T>(string spName, QueryCommand queryCommand, CancellationToken ct)
		{
			var connection = Database.GetDbConnection();
			var commandTimeout = Database.GetCommandTimeout();
			var transaction = Database.CurrentTransaction?.GetDbTransaction();

			var tvpParameters = queryCommand.Parameters.AsTableValuedParameter("SqlVariantTable");

			var command = new CommandDefinition(
				 $"{spName}",
				 new
				 {
					 Command = queryCommand.Command,
					 Parameters = tvpParameters
				 },
				 transaction,
				 commandTimeout,
				 CommandType.StoredProcedure,
				 cancellationToken: ct
			);

			return connection.QueryAsync<T>(command);
		}

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			if (!optionsBuilder.IsConfigured)
			{
				optionsBuilder.UseSqlServer(_sampleConnectionString);
				//optionsBuilder.LogTo((str) => _logger.LogError(str));
			}
		}
	}

	public class UserEntity
	{
		public int Id { get; set; }
		public string Name { get; set; }
		public DateTime? Birth { get; set; }
		public decimal? Price { get; set; }
		public bool IsUat { get; set; }	
	}

	public class QueryCommand
	{
		public string Command { get; set; }
		public List<SqlDataRecord> Parameters { get; set; }
	}
}
