using SqliteCli.Helpers;

namespace SqliteCli.Repos;

public interface IStockService
{
    Task<List<ReportTranItem>> ReportTransAsync();
    Task ShowStockHistoryAsync(ShowStockHistoryReq req);
    void ShowBalance();
    void ShowTransList(ShowTransListCommandLineOptions options);
    Task Test();
}