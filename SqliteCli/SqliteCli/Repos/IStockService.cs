using SqliteCli.Helpers;

namespace SqliteCli.Repos;

public interface IStockService
{
    Task<List<ReportTranItem>> ReportTransAsync();
    Task ShowStockHistoryAsync(ShowStockHistoryReq req);
    void ShowBalance();
    List<TransHistory> GetTransList(ListTransReq listTransReq);
    Task Test();
}