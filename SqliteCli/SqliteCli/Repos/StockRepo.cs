using Dapper;
using Microsoft.EntityFrameworkCore;
using SqliteCli.Entities;
using SqliteCli.Helpers;
using System.Data;
using System.Reflection;
using System.Text.Json;
using T1.Standard.Common;
using T1.Standard.DynamicCode;
using T1.Standard.Web;

namespace SqliteCli.Repos
{
	public class StockRepo : IStockRepo
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

			var q1 = connection.Query(sql, new
			{
				startTime = req.StartDate,
				endTime = req.EndDate,
			});

			var dapperList = q1.ToList();

			var dictList = dapperList.Select(x => (IDictionary<string, object>)x)
				.ToList();


			var list = new List<ReportTranItem>();
			foreach (var dict in dictList)
			{
				var item = dict.ConvertToObject<ReportTranItem>();
				list.Add(item);
			}

			return list;
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

	public interface IStockExchangeApi
	{
		Task<IEnumerable<StockExchangeData>> GetStockTranListAsync(GetStockReq req);
	}

	public class TwseStockExchangeApi : IStockExchangeApi
	{
		static string _baseUrl = "https://www.twse.com.tw";
		private readonly IWebApiClient _webApi;

		public TwseStockExchangeApi(IWebApiClient webApi)
		{
			this._webApi = webApi;
		}

		public async Task<IEnumerable<StockExchangeData>> GetStockTranListAsync(GetStockReq req)
		{
			var date = req.Date.ToString("yyyyMMdd");
			var jsonData = await _webApi.GetAsync(
				$"{_baseUrl}/exchangeReport/STOCK_DAY?response=json&date={date}&stockNo={req.StockId}",
				new Dictionary<string, string>());
			var options = new JsonSerializerOptions
			{
				ReadCommentHandling = JsonCommentHandling.Skip,
				AllowTrailingCommas = true,
				PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
			};
			var rawData = JsonSerializer.Deserialize<StockExchangeRawData>(jsonData, options);
			if (rawData == null)
			{
				return Enumerable.Empty<StockExchangeData>();
			}
			return rawData.GetStockList(req.StockId);
		}
	}

	public class GetStockReq
	{
		public DateTime Date { get; set; }
		public string StockId { get; set; }
	}

	public class StockExchangeData
	{
		[DisplayString("", 10)]
		public DateTime Date { get; set; }
		
		[DisplayString("", 7)]
		public string StockId { get; set; }
		
		[DisplayString("", 7)]
		public long TradeVolume { get; set; }

		[DisplayString("", 7)]
		public decimal DollorVolume { get; set; }
		
		[DisplayString("", 7)]
		public decimal OpeningPrice { get; set; }

		[DisplayString("", 7)]
		public decimal HighestPrice { get; set; }

		[DisplayString("", 7)]
		public decimal LowestPrice { get; set; }
		
		[DisplayString("", 7)]
		public decimal ClosingPrice { get; set; }
		
		[DisplayString("", 7)]
		public decimal Change { get; set; }

		[DisplayString("", 7)]
		public long Transaction { get; set; }
	}

	public class StockExchangeRawData
	{
		public static Dictionary<string, string> FieldNames = new Dictionary<string, string>
		{
			{ "日期", nameof(StockExchangeData.Date) },
			{ "成交股數", nameof(StockExchangeData.TradeVolume) },
			{ "成交金額", nameof(StockExchangeData.DollorVolume) },
			{ "開盤價", nameof(StockExchangeData.OpeningPrice) },
			{ "最高價", nameof(StockExchangeData.HighestPrice) },
			{ "最低價", nameof(StockExchangeData.LowestPrice) },
			{ "收盤價", nameof(StockExchangeData.ClosingPrice) },
			{ "漲跌價差", nameof(StockExchangeData.Change) },
			{ "成交筆數", nameof(StockExchangeData.Transaction) },
		};

		public string Stat { get; set; }
		public string Date { get; set; }
		public string Title { get; set; }
		public List<string> Fields { get; set; }
		public List<List<string>> Data { get; set; }
		public List<string> Notes { get; set; }

		public IEnumerable<StockExchangeData> GetStockList(string stockId)
		{
			var stockTranObjInfo = ReflectionClass.Reflection(typeof(StockExchangeData));
			foreach (var dataItem in Data)
			{
				var stockTran = new StockExchangeData();
				stockTran.StockId = stockId;
				foreach (var item in Fields.Select((value, idx) => new { name = FieldNames[value], idx }))
				{
					var valueStr = dataItem[item.idx];
					var propInfo = stockTranObjInfo.Properties[item.name];
					var value = (object)valueStr;
					if (propInfo.PropertyType != typeof(string))
					{
						if (propInfo.PropertyType.IsValueType)
						{
							valueStr = valueStr.Replace(",", "");
						}
						value = valueStr.ChangeType(propInfo.PropertyType);

						if( propInfo.Name == nameof(StockExchangeData.Date))
						{
							var date = (DateTime)value;
							date = date.AddYears(1911);
							value = date;
						}
					}
					propInfo.Setter(stockTran, value);
				}
				yield return stockTran;
			}
		}
	}
}
