using SqliteCli.Entities;

namespace SqliteCli.Repos
{
	public interface IStockRepo
	{
		void BuyStock(TransEntity data);
		List<ReportTranItem> GetTransGroupByStock(ReportTransReq req);
		void Deposit(DepositReq depositReq);
		void AppendStockHistory(StockHistoryEntity stockHistoryEntity);
		List<StockHistoryEntity> GetStockHistory(GetStockHistoryReq req);
		List<TransEntity> GetStockTranHistory(ShowStockHistoryReq req);
		StockHistoryEntity? GetStockHistoryData(DateTime date, string stockId);
		List<TransHistory> GetTransList(ListTransReq req);
		decimal GetBalance();
		StockHistoryEntity? GetLastStockHistoryData(string stockId);
	}
}
