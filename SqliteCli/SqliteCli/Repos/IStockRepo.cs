using SqliteCli.Entities;

namespace SqliteCli.Repos
{
	public interface IStockRepo
	{
		void BuyStock(TransEntity data);
		List<ReportTranItem> ReportTrans(ReportTransReq req);
		void Deposit(DepositReq depositReq);
		void UpsertStockHistory(StockHistoryEntity stockHistoryEntity);
	}
}
