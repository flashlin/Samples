using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using QueryKits.CsvEx;
using QueryKits.ExcelUtils;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;

public class QueryService : IQueryService
{
    private readonly IReportRepo _reportRepo;

    public QueryService(IReportRepo reportRepo)
    {
        _reportRepo = reportRepo;
    }

    public List<string> GetAllTableNames()
    {
        return _reportRepo.GetAllTableNames();
    }

    public void MergeTable(MergeTableRequest req)
    {
        _reportRepo.MergeTable(req);
    }

    public void ImportCsvFile(string csvFile)
    {
        var csvSheet = CsvSheet.ReadFrom(csvFile, ",");
        var excelSheet = new ExcelSheet();
        foreach (var csvSheetHeader in csvSheet.Headers.Select((value, index) => new {value, index}))
        {
            excelSheet.Headers.Add(new ExcelColumn
            {
                Name = csvSheetHeader.value.Name,
                DataType = ExcelDataType.String,
                CellIndex = csvSheetHeader.index
            });
        }

        excelSheet.Rows.AddRange(csvSheet.Rows);
        var tableName = Path.GetFileNameWithoutExtension(csvFile);
        _reportRepo.ReCreateTable(tableName, excelSheet.Headers);
        _reportRepo.ImportData(tableName, excelSheet);
    }

    public void ImportExcelFile(string xlsxFile)
    {
        var excelSheets = new ExcelHelper().ReadSheets(xlsxFile);
        foreach (var excelSheet in excelSheets)
        {
            var tableName = $"{Path.GetFileNameWithoutExtension(xlsxFile)}_{excelSheet.Name}";
            _reportRepo.ReCreateTable(tableName, excelSheet.Headers);
            _reportRepo.ImportData(tableName, excelSheet);
        }
    }

    public void DeleteTable(string tableName)
    {
        _reportRepo.DeleteTable(tableName);
    }

    public List<ExcelSheet> QueryRawSql(string sql)
    {
        var result = new List<ExcelSheet>();
        var dataSets = _reportRepo.QueryMultipleRawSql(sql);
        foreach (var ds in dataSets)
        {
            if (ds.Rows.Count == 0)
            {
                continue;
            }

            var headers = ds.Rows[0].Keys
                .Select(name => new ExcelColumn
                {
                    Name = name
                }).ToList();
            var sheet = new ExcelSheet();
            sheet.Headers.AddRange(headers);
            foreach (var row in ds.Rows)
            {
                sheet.Rows.Add(row.ToDictionary(x => x.Key, x => $"{x.Value}"));
            }

            result.Add(sheet);
        }

        return result;
    }

    public void AddSqlCode(string sqlCode)
    {
        _reportRepo.AddSqlCode(sqlCode);
    }

    public List<string> GetTop10SqlCode()
    {
        return _reportRepo.GetTop10SqlCode();
    }

    public string ConvertText(string text)
    {
        var textFormat = GetTextFormat(text);
        if (textFormat == TextFormat.JsonArray)
        {
            return text.ToCsvString();
        }

        if (textFormat == TextFormat.Json)
        {
            var jsonStr = $"[{text}]";
            return jsonStr.ToCsvString();
        }

        if (textFormat == TextFormat.Line)
        {
            return ConvertLineToMultipleLine(text);
        }

        var lines = text.Split(Environment.NewLine);
        return string.Join(",", lines);
    }

    private static string ConvertLineToMultipleLine(string text)
    {
        var charGroup = text.GroupBy(x => x)
            .Select(x => new {x.Key, Count = x.Count()})
            .Where(x => new[] {'\t', ','}.Contains(x.Key))
            .ToDictionary(x => x.Key, x => x.Count);
        if (!charGroup.TryGetValue('\t', out var tabCount))
        {
            tabCount = 0;
        }

        if (!charGroup.TryGetValue(',', out var commaCount))
        {
            commaCount = 0;
        }

        var newLine = Environment.NewLine;
        if (tabCount > commaCount)
        {
            return string.Join(newLine, text.Split('\t'));
        }
        return string.Join(newLine, text.Split(','));
    }

    public TextFormat GetTextFormat(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return TextFormat.Empty;
        }

        if (TryReadMultipleLine(text))
        {
            if (TryDeserializeJsonArray(text)) return TextFormat.JsonArray;
            if (TryDeserializeJson(text)) return TextFormat.Json;
            return TextFormat.Text;
        }

        if (TryDeserializeJsonArray(text)) return TextFormat.JsonArrayLine;
        if (TryDeserializeJson(text)) return TextFormat.JsonLine;
        return TextFormat.Line;
    }

    private static bool TryReadMultipleLine(string text)
    {
        var sr = new StringReader(text);
        sr.ReadLine();
        var line2 = sr.ReadLine();
        return line2 != null;
    }

    private static bool TryDeserializeJson(string text)
    {
        try
        {
            var obj = JsonSerializer.Deserialize<dynamic>(text);
            if (obj is JsonElement jsonElement)
            {
                return jsonElement.ValueKind == JsonValueKind.Object;
            }
            return obj != null;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryDeserializeJsonArray(string text)
    {
        try
        {
            var obj = JsonSerializer.Deserialize<List<dynamic>>(text);
            return obj != null;
        }
        catch
        {
            return false;
        }
    }
}

public enum TextFormat
{
    Empty,
    Text,
    Line,
    Json,
    JsonArray,
    JsonArrayLine,
    JsonLine
}