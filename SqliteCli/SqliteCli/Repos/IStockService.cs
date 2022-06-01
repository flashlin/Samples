using SqliteCli.Helpers;

namespace SqliteCli.Repos;

public interface IStockService
{
    Task<List<ReportTranItem>> GetAllStockTransReportAsync();
    Task ShowStockHistoryAsync(ShowStockHistoryReq req);
    void ShowBalance();
    List<TransHistory> GetTransList(ListTransReq listTransReq);
    Task Test();
    Task<List<ReportTranItem>> GetStockReportAsync(ReportTransReq req);
    Task<List<ReportProfitItem>> GetAllStockProfitReportAsync();
    Task<List<TransHistory>> GetOneStockTransAsync(string stockId);
}