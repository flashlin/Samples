using System.Text.Json;
using T1.Standard.Web;

namespace SqliteCli.Repos;

public class FinMindApi : IStockExchangeApi
{
    static string _baseUrl = "https://api.finmindtrade.com";
    private readonly IWebApiClient _webApi;

    public FinMindApi(IWebApiClient webApi)
    {
        _webApi = webApi;
    }

    public async Task<IEnumerable<StockExchangeData>> GetStockTranListAsync(GetStockReq req)
    {
        var url =
            $"{_baseUrl}/api/v4/data?dataset=TaiwanStockPrice&data_id={req.StockId}&start_date={req.StartDate.ToDateString()}&end_date={req.EndDate.ToDateString()}";
        var jsonData = await _webApi.GetAsync(
            url,
            new Dictionary<string, string>());

        var options = new JsonSerializerOptions
        {
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };
        var rawData = JsonSerializer.Deserialize<FinMindResp>(jsonData, options);
        var list = new List<StockExchangeData>();
        foreach (var data in rawData.data)
        {
            list.Add(new StockExchangeData
            {
                Date = data.date,
                StockId = data.stock_id,
                OpeningPrice = data.open,
                ClosingPrice = data.close,
                HighestPrice = data.max,
                LowestPrice = data.min,
                Change = data.spread,
                DollorVolume = data.Trading_money, 
                Transaction = data.Trading_turnover,
                TradeVolume = data.Trading_Volume
            });
        }

        return list;
    }

    public async Task<StockExchangeData> GetLastDataAsync(string stockId)
    {
        var list = await GetStockTranListAsync(new GetStockReq
            {
                StartDate = DateTime.Now, 
                EndDate = DateTime.Now, 
                StockId = stockId
            });
        var data = list.OrderByDescending(x => x.Date).FirstOrDefault();
        if (data == null)
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