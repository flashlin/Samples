using QueryApp.Models.Services;

namespace QueryApp.Models.Helpers;

public class ExcelSheet
{
    public string Name { get; set; } = string.Empty;
    public List<ExcelColumn> Headers { get; set; } = new();
    public List<Dictionary<string, string>> Rows { get; set; } = new();
}