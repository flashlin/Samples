namespace SqliteCli.Repos
{
	public interface IStockExchangeApi
	{
		Task<IEnumerable<StockExchangeData>> GetStockTranListAsync(GetStockReq req);
		Task<StockExchangeData> GetLastDataAsync(string stockId);
	}
}
