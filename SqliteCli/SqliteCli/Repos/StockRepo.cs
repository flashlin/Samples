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

		public void BuyStock(TransEntity data)
		{
			if (data.TranTime == DateTime.MinValue)
			{
				data.TranTime = DateTime.Now;
			}
			data.TranType = "Buy";

			using var db = GetDatabase();

			var stock = db.StocksMap.Where(x => x.Id == data.StockId).FirstOrDefault();
			if (stock == null)
			{
				Console.WriteLine($"Can't found stockId:{data.StockId}");
				return;
			}

			if (data.StockPrice <= 0)
			{
				Console.WriteLine($"Stock price:{data.StockPrice} ERROR");
				return;
			}
			data.HandlingFee = Math.Round(data.StockPrice * data.NumberOfShare * stock.HandlingFee, 0, MidpointRounding.AwayFromZero);
			data.Balance = -(data.StockPrice * data.NumberOfShare + data.HandlingFee);

			db.Trans.Add(data);
			db.SaveChanges();
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

		public List<ReportTranItem> ReportTrans(ReportTransReq req)
		{
			using var db = GetDatabase();

			var sql = @"
select 
    st.Id StockId,
    t.TranType,
    st.StockName,
    MIN(t.StockPrice) minStockPrice,
    AVG(t.StockPrice) avgStockPrice,
    MAX(t.StockPrice) maxStockPrice,
    SUM(t.NumberOfShare) NumberOfShare,
	 SUM(t.HandlingFee) HandlingFee,
    SUM(t.Balance) Balance
from stockMap st 
left join trans t on st.Id = t.StockId
group by st.Id, t.TranType
";

			req.StartDate = DateTime.MinValue;
			req.EndDate = DateTime.Now;

			var connection = db.Database.GetDbConnection();

			var q1 = connection.Query<ReportTranItem>(sql, new
			{
				startTime = req.StartDate,
				endTime = req.EndDate,
			});

			return q1.ToList();
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
						Id = c.tran.Id,
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
