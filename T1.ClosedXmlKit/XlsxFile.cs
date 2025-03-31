using ClosedXML.Excel;

namespace T1.ClosedXmlKit;

public class XlsxFile
{
    private XLWorkbook? _workbook;

    public void Open(string xlsxFile)
    {
        _workbook = new XLWorkbook(xlsxFile);
    }

    public List<Dictionary<string, string>> GetWorksheet(int index)
    {
        if (_workbook == null)
        {
            throw new InvalidOperationException("Please use the Open method to open the Excel file first");
        }

        var worksheet = _workbook.Worksheet(index + 1);
        var range = worksheet.RangeUsed();
        var result = new List<Dictionary<string, string>>();

        // 取得標題列
        var headers = range.FirstRow().CellsUsed()
            .Select(cell => cell.Value.ToString())
            .ToList();

        // 從第二列開始讀取資料
        foreach (var row in range.RowsUsed().Skip(1))
        {
            var rowDict = new Dictionary<string, string>();
            var cells = row.CellsUsed().ToList();

            for (int i = 0; i < Math.Min(headers.Count, cells.Count); i++)
            {
                rowDict[headers[i]] = cells[i].Value.ToString();
            }

            result.Add(rowDict);
        }

        return result;
    }
} 