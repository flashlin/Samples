using CommandLine;
using T1.Standard.Extensions;

namespace SqliteCli;

public class ReportStockCommandLine
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "";

    [Value(index: 1, Required = false, HelpText = "StockId")]
    public string? StockId { get; set; }
}

public class DividendCommandLine
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "di";
    
    [Value(index: 1, Required = true, HelpText = "TranDate")]
    public DateTime TranTime {get; set;}

    [Value(index: 2, Required = true, HelpText = "StockId")]
    public string StockId { get; set; } = string.Empty;
    
    [Value(index: 3, Required = true, HelpText = "股數")]
    public int NumberOfShare { get; set; }
    
    [Value(index: 4, Required = true, HelpText = "股利")]
    public decimal Dividend { get; set; }
}


public class OneStockCommandArgs
{
    [Value(index: 0, HelpText = "action name")]
    public string ActionName { get; set; } = "l";

    [Value(index: 1, Required = true, HelpText = "StockId")]
    public string StockId { get; set; } = String.Empty;
}

public class AddDiscountTranCommandArgs
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
