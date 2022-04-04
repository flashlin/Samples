using Dapper;
using Microsoft.EntityFrameworkCore;
using SqliteCli.Entities;
using SqliteCli.Helpers;
using System.Data;

namespace SqliteCli.Repos
{
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

		public List<TransHistory> ListTrans(ListTransReq req)
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

			var trans = q1.ToList();

			var q2 = trans.GroupJoin(db.StocksMap, tran => tran.StockId, stock => stock.Id,
				(tran, stock) => new
				{
					tran,
					stock
				})
				.SelectMany(
					g => g.stock.DefaultIfEmpty(new StockEntity
					{
						StockName = String.Empty,
						StockType = String.Empty
					}),
					(c, stock) => new TransHistory
					{
						TranTime = c.tran.TranTime,
						TranType = c.tran.TranType,
						StockId = c.tran.StockId,
						StockName = stock.StockName,
						StockPrice = c.tran.StockPrice,
						NumberOfShare = c.tran.NumberOfShare,
						HandlingFee = c.tran.HandlingFee,
						Balance = c.tran.Balance,
					}
				);

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

}
