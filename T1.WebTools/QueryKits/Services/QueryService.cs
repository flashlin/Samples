using System.Text;
using System.Text.Json.Serialization;
using QueryKits.CsvEx;
using QueryKits.ExcelUtils;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;

public class QueryService : IQueryService
{
    private readonly IReportRepo _reportRepo;
    private readonly TextConverter _textConverter;

    public QueryService(IReportRepo reportRepo)
    {
        _reportRepo = reportRepo;
        _textConverter = new TextConverter();
    }

    public TextConverter TextConverter
    {
        get { return _textConverter; }
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
        return _textConverter.ConvertText(text);
    }
}