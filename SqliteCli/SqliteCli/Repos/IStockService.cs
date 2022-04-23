namespace SqliteCli.Repos;

public interface IStockService
{
    Task<List<ReportTranItem>> ReportTransAsync();
}