namespace SqliteCli.Repos
{
	public interface IStockExchangeApi
	{
		IAsyncEnumerable<StockExchangeData> GetStockHistoryListAsync(GetStockReq req);
		Task<StockExchangeData> GetLastDataAsync(string stockId);
	}
}
