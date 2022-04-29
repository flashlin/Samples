using System.Data;
using System.Text.Json;
using SqliteCli.Helpers;
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

        public async IAsyncEnumerable<StockExchangeData> GetStockHistoryListAsync(GetStockReq req)
        {
            foreach (var month in req.DateRange.GetRangeByMonth())
            {
                var monthStr = month.ToString("yyyyMMdd");
                var result = await GetStockTranListAsync(req, monthStr);
                foreach (var data in result)
                {
                    yield return data;
                }
            }
        }

        private async Task<IEnumerable<StockExchangeData>> GetStockTranListAsync(GetStockReq req, string date)
        {
            var url = $"{_baseUrl}/exchangeReport/STOCK_DAY?response=json&date={date}&stockNo={req.StockId}";
            var jsonData = await _webApi.GetAsync(
                url,
                new Dictionary<string, string>());
            var options = new JsonSerializerOptions
            {
                ReadCommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            };
            try
            {
                var rawData = JsonSerializer.Deserialize<StockExchangeRawData>(jsonData, options);
                if (rawData == null)
                {
                    return Enumerable.Empty<StockExchangeData>();
                }

                return rawData.GetStockList(req.StockId);
            }
            catch
            {
                //Console.WriteLine($"{jsonData}");
                return Enumerable.Empty<StockExchangeData>();
            }
        }
    }
}