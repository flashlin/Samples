using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryApp.Models.Services;
using QueryKits.ExcelUtils;
using QueryKits.Extensions;
using QueryKits.Services;
using T1.WebTools.CsvEx;

namespace QueryApp.Controllers.Apis;

[ApiController]
[Route("api/[controller]/[action]")]
public class LocalQueryApiController : ControllerBase
{
    private readonly IReportRepo _reportRepo;
    private readonly ILocalEnvironment _localEnvironment;

    public LocalQueryApiController(IReportRepo reportRepo, ILocalEnvironment localEnvironment)
    {
        _localEnvironment = localEnvironment;
        _reportRepo = reportRepo;
    }
    
    [HttpPost]
    public KnockResponse Knock(KnockRequest req)
    {
        if (_localEnvironment.AppUid != req.AppUid)
        {
            return new KnockResponse()
            {
                IsSuccess = false
            };
        }

        _localEnvironment.UserUid = req.UniqueId;
        _localEnvironment.LastActivityTime = DateTime.Now;
        _localEnvironment.IsBinded = true;
        return new KnockResponse
        {
            IsSuccess = true,
            AppVersion = _localEnvironment.AppVersion
        };
    }

    [HttpPost]
    public GetAllTableNamesResponse GetAllTableNames()
    {
        return new GetAllTableNamesResponse()
        {
            TableNames = _reportRepo.GetAllTableNames()
        };
    }
    
    [HttpPost]
    public async Task UploadFiles()
    {
        var dataFolder = Path.Combine(_localEnvironment.AppLocation, "Data");
        DirectoryHelper.EnsureDirectory(dataFolder);
        
        var uploadFiles = this.Request.Form.Files;
        foreach (var uploadFile in uploadFiles)
        {
            if (uploadFile.Length == 0)
                continue;
            
            var fileName = Path.GetFileName(uploadFile.FileName);
            var fileExt = GetFileExtName(fileName);
            if (!IsValidFileExt(fileExt))
            {
                continue;
            }
            var fileSize = uploadFile.Length;
            var file = Path.Combine(dataFolder, fileName);
            if (System.IO.File.Exists(file))
            {
                System.IO.File.Delete(file);
            }
            
            await using var stream = new FileStream(file, FileMode.Create);
            await uploadFile.CopyToAsync(stream);
            await stream.FlushAsync();
            stream.Close();

            if (fileExt == "xlsx")
            {
                var excelSheets = new ExcelHelper().ReadSheets(file);
                foreach (var excelSheet in excelSheets)
                {
                    var tableName = $"{Path.GetFileNameWithoutExtension(fileName)}_{excelSheet.Name}";
                    _reportRepo.ReCreateTable(tableName, excelSheet.Headers);
                    _reportRepo.ImportData(tableName, excelSheet);
                }
                continue;
            }

            if (fileExt == "csv")
            {
                ImportLocalCsvFile(file);       
                continue;
            }
        }
    }

    private static string GetFileExtName(string fileName)
    {
        var fileExt = fileName.Substring(Path.GetFileNameWithoutExtension(fileName).Length);
        if (fileExt.StartsWith("."))
        {
            fileExt = fileExt.Substring(1);
        }

        return fileExt;
    }

    [HttpPost]
    public OkResult ImportLocalFile(ImportLocalFileRequest req)
    {
        var extName = GetFileExtName(Path.GetFileName(req.FilePath));
        if (extName == "csv")
        {
            ImportLocalCsvFile(req.FilePath);
            return Ok();
        }
        ImportLocalJsonFile(req.FilePath);
        return Ok();
    }

    public void ImportLocalJsonFile(string jsonFile)
    {
        var csvFileName = Path.GetFileName(jsonFile);
        csvFileName = csvFileName.Substring(0, Path.GetFileNameWithoutExtension(csvFileName).Length) + ".csv";
        var folder = Path.GetDirectoryName(jsonFile) ?? "";
        var csvFile = Path.Combine(folder, csvFileName);
        var json = System.IO.File.ReadAllText(jsonFile);
        var dictList = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(json)!;
        dictList.ToCsvFile(csvFile);
        ImportLocalCsvFile(csvFile);
    }

    private void ImportLocalCsvFile(string file)
    {
        var delimiter = CsvSheet.ParseHeaderDelimiterFromFile(file);
        var csv = CsvSheet.ReadFrom(file, delimiter);
        var tableName = Path.GetFileNameWithoutExtension(file);
        var excelSheet = new ExcelSheet
        {
            Headers = csv.Headers.Select((x, index) => new ExcelColumn
            {
                Name = x.Name,
                DataType = ExcelDataType.String,
                CellIndex = index
            }).ToList(),
            Rows = csv.Rows
        };
        _reportRepo.ReCreateTable(tableName, excelSheet.Headers);
        _reportRepo.ImportData(tableName, excelSheet);
    }

    private static bool IsValidFileExt(string fileExt)
    {
        var invalidFileExts = new[] { "csv", "xlsx" };
        return invalidFileExts.Contains(fileExt);
    }

    [HttpPost]
    public QueryRawSqlResponse QueryRawSql(QueryRawSqlRequest req)
    {
        try
        {
            var dataList = _reportRepo.QueryRawSql(req.Sql)
                .ToList();

            if (dataList.Count == 0)
            {
                return new QueryRawSqlResponse()
                {
                    ErrorMessage = "No Data Found."
                };
            }

            var csvSheet = dataList.ToCsvStream().ToCsvSheet();
            csvSheet.SaveToFile(_localEnvironment.AppLocation + "/Data/Result.csv");
            return new QueryRawSqlResponse
            {
                CsvSheet = csvSheet,
            };
        }
        catch(Exception e)
        {
            return new QueryRawSqlResponse
            {
                ErrorMessage = e.Message
            };
        }
    }
}

public class ImportLocalFileRequest
{
    public string FilePath { get; set; } = string.Empty;
}

public class GetAllTableNamesResponse
{
    public List<string> TableNames { get; set; } = new();
}

public class QueryRawSqlRequest
{
    public string Sql { get; set; } = string.Empty;
}