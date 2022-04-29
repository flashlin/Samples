using System.Text.Json;
using T1.Standard.Web;

namespace SqliteCli.Repos;

public class FinMindApi : IStockExchangeApi
{
    static string _baseUrl = "https://api.finmindtrade.com";
    private readonly IWebApiClient _webApi;
    private string _token = string.Empty;

    public FinMindApi(IWebApiClient webApi)
    {
        _webApi = webApi;
        //_token = File.ReadAllText(@"D:/VDisk/SNL/finmind.key");
    }

    public async IAsyncEnumerable<StockExchangeData> GetStockHistoryListAsync(GetStockReq req)
    {
        var startDateStr = req.DateRange.StartDate.ToDateString();
        var endDateStr = req.DateRange.EndDate.ToDateString();
        
        var url =
            $"{_baseUrl}/api/v4/data?dataset=TaiwanStockPrice&data_id={req.StockId}&start_date={startDateStr}&end_date={endDateStr}&token={_token}";
        var jsonData = await _webApi.GetAsync(
            url,
            new Dictionary<string, string>());

        if (jsonData == null)
        {
            yield break;
        }

        var options = new JsonSerializerOptions
        {
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        };
        var rawData = JsonSerializer.Deserialize<FinMindResp>(jsonData, options);
        foreach (var data in rawData.data)
        {
            yield return new StockExchangeData
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
            };
        }
    }
}