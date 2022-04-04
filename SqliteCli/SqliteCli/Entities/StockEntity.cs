using Dapper;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage;
using SqliteCli.Helpers;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Data;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SqliteCli.Entities
{
	public enum StockType
	{
		Bond,
		Price_ETF,
		Dividend_ETF,
		Stock,
	}

	[Table("StockMap")]
	public class StockEntity
	{
		[Key]
		[StringLength(20)]
		public string Id { get; set; }

		[StringLength(100)]
		public string StockName { get; set; }
		public StockType StockType { get; set; }
		public decimal HandlingFee { get; set; }
	}

	public class TransEntity
	{
		[Key]
		public long Id { get; set; }
		public DateTime TranTime { get; set; }
		public string TranType { get; set; }
		public string StockId { get; set; }
		public decimal StockPrice { get; set; }
		public int NumberOfShare { get; set; }
		public decimal HandlingFee { get; set; }
		public decimal Balance { get; set; }
	}

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

	public class TransHistory
	{
		public DateTime TranTime { get; set; }
		public string TranType { get; set; }
		public string StockId { set; get; }
		public string StockName { set; get; }
		public decimal StockPrice { get; set; }
		public int NumberOfShare { get; set; }
		public decimal HandlingFee { get; set; }
		public decimal Balance { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{TranTime.ToString("yyyy/MM/dd")}");
			sb.Append($" {TranType}");
			sb.Append($" {StockId}");
			sb.Append($" {StockName}");
			sb.Append($" {StockPrice}");
			sb.Append($" {NumberOfShare}");
			sb.Append($" {HandlingFee}");
			sb.Append($" {Balance}");
			return sb.ToString();
		}
	}

	public class ListTransReq
	{
		public DateTime? StartTime { get; set; }
		public DateTime? EndTime { get; set; }
		public string? StockId { get; set; }
	}

	public class StockRepo
	{
		public IEnumerable<TransHistory> QueryTrans(string cmd)
		{
			using var stockDb = GetDatabase();
			//stockDb.Trans.FromSqlRaw(cmd);
			using var connection = stockDb.Database.GetDbConnection();
			return connection.Query<TransHistory>(cmd);
		}

		public List<TransHistory> ListTrans3(ListTransReq req)
		{
			using var db = GetDatabase();

			var queryFilter = new List<QueryableFilter>();
			if (req.StartTime != null)
			{
				queryFilter.Add(
					new QueryableFilter
					{
						Name = nameof(TransEntity.TranTime),
						Value = req.StartTime,
						Compare = QueryableFilterCompareEnum.GreaterThanOrEqual
					}
				);
			}

			if (req.EndTime != null)
			{
				queryFilter.Add(
					new QueryableFilter
					{
						Name = nameof(TransEntity.TranTime),
						Value = req.EndTime,
						Compare = QueryableFilterCompareEnum.LessThanOrEqual
					}
				);
			}

			var trans = new DynamicFilters<TransEntity>(db)
				.Filter(queryFilter)
				.ToList();

			var q2 = trans.Join(db.StocksMap, tran => tran.StockId, stock => stock.Id,
				(tran, stock) => new TransHistory
				{
					TranTime = tran.TranTime,
					TranType = tran.TranType,
					StockId = tran.StockId,
					StockName = stock.StockName,
					StockPrice = tran.StockPrice,
					NumberOfShare = tran.NumberOfShare,
					HandlingFee = tran.HandlingFee,
					Balance = tran.Balance,
				});

			return q2.ToList();
		}

		public List<TransHistory> ListTrans2(ListTransReq req)
		{
			using var db = GetDatabase();

			var q1 = db.Trans.AsQueryable();

			if (req.StartTime != null)
			{
				q1 = q1.Where(x => x.TranTime >= req.StartTime);
			}

			if (req.EndTime != null)
			{
				q1 = q1.Where(x => x.TranTime <= req.EndTime);
			}
			//.WhereDynamic(x => x.TranTime <= req.StartTime);

			var code = q1.ToQueryString();

			var trans = q1.ToList();

			var q2 = trans.Join(db.StocksMap, tran => tran.StockId, stock => stock.Id,
				(tran, stock) => new TransHistory
				{
					TranTime = tran.TranTime,
					TranType = tran.TranType,
					StockId = tran.StockId,
					StockName = stock.StockName,
					StockPrice = tran.StockPrice,
					NumberOfShare = tran.NumberOfShare,
					HandlingFee = tran.HandlingFee,
					Balance = tran.Balance,
				});

			return q2.ToList();
		}

		public IEnumerable<TransHistory> ListTrans1(ListTransReq req)
		{
			using var db = GetDatabase();

			var queryFilter = new List<QueryableFilter>();
			if (req.StartTime != null)
			{
				queryFilter.Add(
					new QueryableFilter
					{
						Name = nameof(TransEntity.TranTime),
						Value = req.StartTime,
						Compare = QueryableFilterCompareEnum.GreaterThanOrEqual
					}
				);
			}

			if (req.EndTime != null)
			{
				queryFilter.Add(
					new QueryableFilter
					{
						Name = nameof(TransEntity.TranTime),
						Value = req.EndTime,
						Compare = QueryableFilterCompareEnum.LessThanOrEqual
					}
				);
			}

			var query = new DynamicFilters<TransEntity>(db)
				.Filter(queryFilter)
				.Join(db.StocksMap, tran => tran.StockId, stock => stock.Id,
				(tran, stock) => new TransHistory
				{
					TranTime = tran.TranTime,
					TranType = tran.TranType,
					StockId = tran.StockId,
					StockName = stock.StockName,
					StockPrice = tran.StockPrice,
					NumberOfShare = tran.NumberOfShare,
					HandlingFee = tran.HandlingFee,
					Balance = tran.Balance,
				});

			return query;
		}

		protected StockDatabase GetDatabase()
		{
			return new StockDatabase("d:/VDisk/SNL/flash_stock.db");
		}
	}

	public static class DbContextExtension
	{
		public static IEnumerable<T> FromSqlQuery<T>(this DbContext context, string query,
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
