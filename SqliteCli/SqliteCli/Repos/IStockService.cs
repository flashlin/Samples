using SqliteCli.Helpers;

namespace SqliteCli.Repos;

public interface IStockService
{
    Task<List<ReportTranItem>> ReportTransAsync();
    Task AppendStockHistoryAsync(DateRange dateRange, string stockId);
}