namespace SqliteCli.Repos
{
	public interface IStockExchangeApi
	{
		Task<IEnumerable<StockExchangeData>> GetStockHistoryListAsync(GetStockReq req);
		Task<StockExchangeData> GetLastDataAsync(string stockId);
	}
}
