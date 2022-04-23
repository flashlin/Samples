using SqliteCli.Entities;

namespace SqliteCli.Repos
{
	public interface IStockRepo
	{
		void BuyStock(TransEntity data);
		List<ReportTranItem> ReportTrans(ReportTransReq req);
		void Deposit(DepositReq depositReq);
		void UpsertStockHistory(StockHistoryEntity stockHistoryEntity);
		List<StockHistoryEntity> GetStockHistory(GetStockHistoryReq req);
		List<TransEntity> GetStockTranHistory(StockReportHistoryReq req);
	}
}
