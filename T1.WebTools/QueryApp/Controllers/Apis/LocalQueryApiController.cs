using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryApp.Models.Services;

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
            IsSuccess = true
        };
    }

    [HttpPost]
    public List<string> GetAllTableNames()
    {
        return _reportRepo.GetAllTableNames();
    }
    
    [HttpPost]
    public async Task UploadFiles()
    {
        var dataFolder = Path.Combine(_localEnvironment.AppLocation, "Data");
        if (!Directory.Exists(dataFolder))
        {
            Directory.CreateDirectory(dataFolder);
        }
        
        var uploadFiles = this.Request.Form.Files;
        foreach (var uploadFile in uploadFiles)
        {
            if (uploadFile.Length == 0)
                continue;
            var fileName = Path.GetFileName(uploadFile.FileName);
            var fileSize = uploadFile.Length;
            var fileExt = fileName.Substring(Path.GetFileNameWithoutExtension(fileName).Length);

            var file = Path.Combine(dataFolder, fileName);
            if (System.IO.File.Exists(file))
            {
                System.IO.File.Delete(file);
            }
            await using var stream = new FileStream(file, FileMode.Create);
            await uploadFile.CopyToAsync(stream);
            await stream.FlushAsync();
        }
    }
    
    [HttpPost]
    public QueryRawSqlResponse QueryRawSql(QueryRawSqlRequest req)
    {
        try
        {
            var data = _reportRepo.QueryRawSql(req.Sql)
                .Select(row => row.ToDictionary(item => item.Key, y => $"{y.Value}"))
                .ToList();
            return new QueryRawSqlResponse
            {
                Data = data
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

public class QueryRawSqlRequest
{
    public string Sql { get; set; } = string.Empty;
}