using CommandLine;

namespace SqliteCli;

public class ReportStockCommand
{
    [Value(index: 0, HelpText = "actio name")]
    public string ActionName { get; set; }

    [Value(index: 1, Required = false, HelpText = "StockId")]
    public string? StockId { get; set; }
}