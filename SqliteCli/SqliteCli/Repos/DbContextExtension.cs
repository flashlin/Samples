using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage;
using System.Data;
using System.Data.Common;

namespace SqliteCli.Repos
{
	public static class DbContextExtension
	{
		public static IEnumerable<T> FromSqlQuery<T>(this DbContext context,
			string query,
			Func<DbDataReader, T> map, params object[] parameters)
		{
			using (var command = context.Database.GetDbConnection().CreateCommand())
			{
				if (command.Connection.State != ConnectionState.Open)
				{
					command.Connection.Open();
				}
				var currentTransaction = context.Database.CurrentTransaction;
				if (currentTransaction != null)
				{
					command.Transaction = currentTransaction.GetDbTransaction();
				}
				command.CommandText = query;
				if (parameters.Any())
				{
					command.Parameters.AddRange(parameters);
				}
				using (var result = command.ExecuteReader())
				{
					while (result.Read())
					{
						yield return map(result);
					}
				}
			}
		}

		public static DbTransaction GetDbTransaction(this IDbContextTransaction source)
		{
			return (source as IInfrastructure<DbTransaction>).Instance;
		}
	}

}
