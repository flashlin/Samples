using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using NPOI.OpenXml4Net.OPC.Internal;
using QueryKits.Extensions;
using QueryKits.Services;
using QueryWeb.Models;

namespace QueryWeb.Controllers.api;

[ApiController]
[Route("api/[controller]/[action]")]
public class FileController : ControllerBase
{
    private readonly IQueryEnvironment _queryEnvironment;
    private IQueryService _queryService;

    public FileController(IQueryEnvironment queryEnvironment, IQueryService queryService)
    {
        _queryEnvironment = queryEnvironment;
        _queryService = queryService;
    }

    [HttpPost]
    public string SayHello([FromBody] string name)
    {
        return $"Hello {name}";
    }
    
    [HttpPost]
    public IActionResult Upload([FromForm] UploadFilesRequest req)
    {
        var fileName = Path.GetFileName(req.FileName);
        DirectoryHelper.EnsureDirectory(_queryEnvironment.UploadPath);
        var fullFilename = Path.Combine(_queryEnvironment.UploadPath, fileName);
        using var file = new FileStream(fullFilename, FileMode.OpenOrCreate);
        file.Seek(req.CurrentChunk * 2048, SeekOrigin.Begin);
        var buffer = new byte[2048];
        var count = req.Chunk.OpenReadStream().Read(buffer, 0, 2048);
        file.Write(buffer, 0, count);
        file.Flush();
        return Ok();
    }
}