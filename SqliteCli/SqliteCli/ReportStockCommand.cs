using CommandLine;

namespace SqliteCli;

public class ReportStockCommand
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";

    [Value(index: 1, Required = false, HelpText = "StockId")]
    public string? StockId { get; set; }
}

public class AddDiscountTranCommand
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";
    
    [Value(index: 1, Required = false, HelpText = "TranDate")]
    public DateTime? TranDate { get; set; }
    
    [Value(index: 2, Required = false, HelpText = "amount")]
    public decimal Amount { get; set; }

    [Value(index: 3, Required = false, HelpText = "remark")]
    public string Remark { get; set; } = "";
}
