namespace SqliteCli.Repos;

public class FinMindStockData
{
    public DateTime date { get; set; }
    public string stock_id { get; set; }
    public long Trading_Volume { get; set; }
    public long  Trading_money { get; set; }
    public decimal open { get; set; }
    public decimal max { get; set; }
    public decimal min { get; set; }
    public decimal close { get; set; }
    public decimal spread { get; set; }
    public long Trading_turnover { get; set; }
}