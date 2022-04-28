namespace SqliteCli.Repos;

public class FinMindResp
{
    public string msg { get; set; }
    public int status{get; set;}
    public List<FinMindStockData> data { get; set; }
}