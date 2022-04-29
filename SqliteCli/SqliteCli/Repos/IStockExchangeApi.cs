namespace SqliteCli.Repos
{
	public interface IStockExchangeApi
	{
		IAsyncEnumerable<StockExchangeData> GetStockHistoryListAsync(GetStockReq req);
	}
}
