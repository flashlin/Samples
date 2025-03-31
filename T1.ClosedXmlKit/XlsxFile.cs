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

    public IXLWorksheet Compare(List<Dictionary<string, string>> sheet1, List<Dictionary<string, string>> sheet2)
    {
        var workbook = new XLWorkbook();
        var resultSheet = workbook.Worksheets.Add("Compare");

        // 取得所有欄位名稱
        var allFields = sheet1.SelectMany(d => d.Keys)
            .Concat(sheet2.SelectMany(d => d.Keys))
            .Distinct()
            .ToList();

        // 寫入標題列
        for (int i = 0; i < allFields.Count; i++)
        {
            resultSheet.Cell(1, i + 1).Value = allFields[i];
        }

        int currentRow = 2;

        // 建立索引以加速查找
        var sheet2Dict = sheet2.ToDictionary(
            row => string.Join("|", row.OrderBy(x => x.Key).Select(x => $"{x.Key}={x.Value}")),
            row => row
        );

        // 比對 sheet1 的每一列
        foreach (var row1 in sheet1)
        {
            var row1Key = string.Join("|", row1.OrderBy(x => x.Key).Select(x => $"{x.Key}={x.Value}"));
            
            if (sheet2Dict.TryGetValue(row1Key, out var matchingRow2))
            {
                // 完全相同的列，直接複製
                for (int i = 0; i < allFields.Count; i++)
                {
                    resultSheet.Cell(currentRow, i + 1).Value = row1.GetValueOrDefault(allFields[i], "");
                }
            }
            else
            {
                // 檢查是否有部分相同的列
                bool foundPartialMatch = false;
                foreach (var row2 in sheet2)
                {
                    if (row1.Keys.Intersect(row2.Keys).Any())
                    {
                        // 找到部分匹配的列，標記差異
                        for (int i = 0; i < allFields.Count; i++)
                        {
                            var field = allFields[i];
                            var value1 = row1.GetValueOrDefault(field, "");
                            var value2 = row2.GetValueOrDefault(field, "");
                            
                            var cell = resultSheet.Cell(currentRow, i + 1);
                            cell.Value = value1;
                            
                            if (value1 != value2)
                            {
                                cell.Style.Fill.BackgroundColor = XLColor.Red;
                            }
                        }
                        foundPartialMatch = true;
                        break;
                    }
                }

                if (!foundPartialMatch)
                {
                    // sheet1 獨有的列，使用黃色背景
                    for (int i = 0; i < allFields.Count; i++)
                    {
                        var cell = resultSheet.Cell(currentRow, i + 1);
                        cell.Value = row1.GetValueOrDefault(allFields[i], "");
                        cell.Style.Fill.BackgroundColor = XLColor.Yellow;
                    }
                }
            }
            currentRow++;
        }

        // 檢查 sheet2 獨有的列
        foreach (var row2 in sheet2)
        {
            var row2Key = string.Join("|", row2.OrderBy(x => x.Key).Select(x => $"{x.Key}={x.Value}"));
            if (!sheet1.Any(row1 => 
                string.Join("|", row1.OrderBy(x => x.Key).Select(x => $"{x.Key}={x.Value}")) == row2Key))
            {
                // sheet2 獨有的列，使用灰色背景
                for (int i = 0; i < allFields.Count; i++)
                {
                    var cell = resultSheet.Cell(currentRow, i + 1);
                    cell.Value = row2.GetValueOrDefault(allFields[i], "");
                    cell.Style.Fill.BackgroundColor = XLColor.Gray;
                }
                currentRow++;
            }
        }

        // 自動調整欄寬
        resultSheet.Columns().AdjustToContents();

        return resultSheet;
    }
}