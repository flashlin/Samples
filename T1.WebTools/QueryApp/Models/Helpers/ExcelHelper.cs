using NPOI.SS.UserModel;
using NPOI.XSSF.UserModel;

namespace QueryApp.Models.Helpers;

public class ExcelHelper
{
    public IEnumerable<ExcelSheet> ReadSheets(string xlsxFile)
    {
        using var stream = new FileStream(xlsxFile, FileMode.Open);
        using var workbook = new XSSFWorkbook(stream);
        for (var i = 0; i < workbook.NumberOfSheets; i++)
        {
            yield return ReadSheet(workbook.GetSheetAt(i));
        }
    }

    private ExcelSheet ReadSheet(ISheet sheet)
    {
        var result = new ExcelSheet
        {
            Name = sheet.SheetName,
            Header = QueryHeader(sheet).ToList()
        };
        for (var i = sheet.FirstRowNum + 1; i < sheet.LastRowNum; i++)
        {
            var row = sheet.GetRow(i);
            if (row == null)
            {
                continue;
            }
            if (row.Cells.All(d => d.CellType == CellType.Blank))
            {
                continue;
            }
            FetchRowData(row, result.Header);
        }
        return result;
    }

    private static Dictionary<string, string> FetchRowData(IRow row, IEnumerable<ExcelColumn> headerNames)
    {
        var rowData = new Dictionary<string, string>();
        foreach (var header in headerNames)
        {
            var cell = row.GetCell(header.CellIndex);
            if (cell == null)
            {
                rowData[header.Name] = string.Empty;
                continue;
            }
            var cellContent = cell.ToString();
            if (string.IsNullOrEmpty(cellContent) || string.IsNullOrWhiteSpace(cellContent))
            {
                rowData[header.Name] = string.Empty;
                continue;
            }
            rowData[header.Name] = cellContent;
        }
        return rowData;
    }

    private static IEnumerable<ExcelColumn> QueryHeader(ISheet sheet)
    {
        var headerRow = sheet.GetRow(0);
        for (var i = 0; i < headerRow.LastCellNum; i++)
        {
            var cell = headerRow.GetCell(i);
            if (cell == null)
            {
                continue;
            }

            var cellContent = cell.ToString();
            if (string.IsNullOrWhiteSpace(cellContent))
            {
                continue;
            }

            yield return new ExcelColumn
            {
                CellIndex = i,
                Name = cellContent
            };
        }
    }
}