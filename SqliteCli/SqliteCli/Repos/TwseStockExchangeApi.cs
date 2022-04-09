using System.Data;
using System.Text.Json;
using T1.Standard.Web;

namespace SqliteCli.Repos
{
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
			string url = $"{_baseUrl}/exchangeReport/STOCK_DAY?response=json&date={date}&stockNo={req.StockId}";
			var jsonData = await _webApi.GetAsync(
				url,
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

		public async Task<StockExchangeData> GetLastDataAsync(string stockId)
		{
			var list = await GetStockTranListAsync(new GetStockReq { Date = DateTime.Now.AddDays(-1), StockId = stockId });
			var data = list.OrderByDescending(x => x.Date).FirstOrDefault();
			if( data== null)
			{
				return new StockExchangeData()
				{
					Date = DateTime.Now, 
					StockId = stockId,
				};
			}
			return data;
		}
	}
}
