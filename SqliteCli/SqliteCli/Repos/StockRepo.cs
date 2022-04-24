using Dapper;
using Microsoft.EntityFrameworkCore;
using SqliteCli.Entities;
using System.Data;
using System.Reflection;

namespace SqliteCli.Repos
{
    public class StockRepo : IStockRepo
    {
        private readonly StockDbContext _db;

        public StockRepo(StockDbContext db)
        {
            _db = db;
        }

        public IEnumerable<TransHistory> QueryTrans(string cmd)
        {
            //using var stockDb = GetDatabase();
            var stockDb = _db;
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

            /*
            using var db = GetDatabase();
            */
            var db = _db;

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

            data.HandlingFee = Math.Round(data.StockPrice * data.NumberOfShare * stock.HandlingFee, 0,
                MidpointRounding.AwayFromZero);
            data.Balance = -(data.StockPrice * data.NumberOfShare + data.HandlingFee);

            db.Trans.Add(data);
            db.SaveChanges();
        }

        public List<ReportTranItem> ReportTrans(ReportTransReq req)
        {
            var sql = @"
select 
    st.Id StockId,
    t.TranType,
    st.StockName,
	 MIN(t.tranTime) minTranTime,
	 MAX(t.tranTime) maxTranTime,
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

            var list = QueryRaw<ReportTranItem>(sql, req).ToList();
            return list;
        }

        public List<StockHistoryEntity> GetStockHistory(GetStockHistoryReq req)
        {
            //using var db = GetDatabase();
            var db = _db;
            var data = db.StocksHistory.Where(x =>
                x.TranDate >= req.StartTime && x.TranDate <= req.EndTime && x.StockId == req.StockId);
            return data.ToList();
        }

        public List<TransEntity> GetStockTranHistory(StockReportHistoryReq req)
        {
            //using var db = GetDatabase();
            var db = _db;
            var data = db.Trans.Where(x => x.TranTime >= req.StartTime && x.TranTime <= req.EndTime
                                                                       && x.StockId == req.StockId
                                                                       && x.TranType == "Buy");
            return data.ToList();
        }

        public StockHistoryEntity? GetStockHistoryData(DateTime date, string stockId)
        {
            //using var db = GetDatabase();
            var db = _db;
            return db.StocksHistory
                .FirstOrDefault(x => x.TranDate == date.Date && x.StockId == stockId);
        }

        public List<TransHistory> ListTrans(ListTransReq req)
        {
            //using var db = GetDatabase();
            var db = _db;

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

            return q2.OrderBy(x => x.TranTime).ToList();
        }

        /*
        protected StockDatabase GetDatabase()
        {
            return new StockDatabase("d:/VDisk/SNL/flash_stock.db");
        }
        */

        public void Deposit(DepositReq depositReq)
        {
            //using var db = GetDatabase();
            var db = _db;
            db.Trans.Add(new TransEntity
            {
                TranTime = depositReq.TranTime,
                TranType = "Deposit",
                StockId = string.Empty,
                Balance = depositReq.Balance,
            });
            db.SaveChanges();
        }

        public void AppendStockHistory(StockHistoryEntity stockHistoryEntity)
        {
            if (stockHistoryEntity.OpeningPrice == 0 && stockHistoryEntity.ClosingPrice == 0)
            {
                return;
            }

            var sql =
                @"insert into stockHistory(TranDate, StockId, TradeVolume, DollorVolume, OpeningPrice, ClosingPrice, HighestPrice, LowestPrice, TransactionCount)
select @TranDate, @StockId, @TradeVolume, @DollorVolume, @OpeningPrice, @ClosingPrice, @HighestPrice, @LowestPrice, @TransactionCount
where not exists( 
    select 1 from stockHistory 
    where DATE(tranDate)=DATE(@TranDate) and stockId=@StockId
    LIMIT 1
)";

            //ExecuteRaw(sql, stockHistoryEntity);
            var exists = _db.StocksHistory.Any(x =>
                x.TranDate == stockHistoryEntity.TranDate && x.StockId == stockHistoryEntity.StockId);
            if (!exists)
            {
                Console.WriteLine($"add {stockHistoryEntity.TranDate} {stockHistoryEntity.StockId}");
                _db.StocksHistory.Add(stockHistoryEntity);
                _db.SaveChanges();
            }
        }

        protected IEnumerable<T> QueryRaw<T>(string sql, object queryParameter)
            where T : class, new()
        {
            //using var db = GetDatabase();
            var db = _db;
            var connection = db.Database.GetDbConnection();
            var q1 = connection.Query(sql, queryParameter);

            var dapperList = q1.ToList();
            var dictList = dapperList.Select(x => (IDictionary<string, object>) x);
            foreach (var dict in dictList)
            {
                var item = dict.ConvertToObject<T>();
                yield return item;
            }
        }

        protected void ExecuteRaw(string sql, object queryParameter)
        {
            //using var db = GetDatabase();
            var db = _db;
            var connection = db.Database.GetDbConnection();
            connection.Execute(sql, queryParameter);
        }
    }

    public class GetStockHistoryReq
    {
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public string StockId { get; set; }
    }
}