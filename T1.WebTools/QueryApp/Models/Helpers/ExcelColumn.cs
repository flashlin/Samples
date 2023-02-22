using QueryApp.Models.Services;

namespace QueryApp.Models.Helpers;

public class ExcelColumn
{
    public string Name { get; set; } = string.Empty;
    public ExcelDataType DataType { get; set; }
    public int CellIndex { get; set; }
}