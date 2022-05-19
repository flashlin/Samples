using CommandLine;

namespace SqliteCli;

public class BuyStockTranCommandLine
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";
    
    [Value(index: 1, Required = true, HelpText = "TranDate")]
    public DateTime TranDate { get; set; } = DateTime.Now;
    
    [Value(index: 2, Required = true, HelpText = "StockId")]
    public string StockId { get; set; }
    
    [Value(index: 3, Required = true, HelpText = "Stock Price")]
    public decimal StockPrice { get; set; }
    
    [Value(index: 4, Required = true, HelpText = "Stock Number Of Share")]
    public int NumberOfShare { get; set; }
}