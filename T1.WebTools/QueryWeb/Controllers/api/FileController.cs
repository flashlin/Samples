using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using NPOI.OpenXml4Net.OPC.Internal;
using QueryKits.Extensions;

namespace QueryWeb.Controllers.api;

[ApiController]
[Route("api/[controller]/[action]")]
public class FileController : ControllerBase
{
    private IWebHostEnvironment _webHostEnv;

    public FileController(IWebHostEnvironment webHostEnv)
    {
        _webHostEnv = webHostEnv;
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
        var uploadPath = Path.Combine(_webHostEnv.WebRootPath, "Upload");
        DirectoryHelper.EnsureDirectory(uploadPath);
        using var file = new FileStream(Path.Combine(uploadPath, fileName), FileMode.OpenOrCreate);
        file.Seek(req.CurrentChunk * 2048, SeekOrigin.Begin);
        var buffer = new byte[2048];
        var count = req.Chunk.OpenReadStream().Read(buffer, 0, 2048);
        file.Write(buffer, 0, count);
        file.Flush();
        return Ok();
    }
}

public class UploadFilesRequest
{
    public string FileName { get; set; } = string.Empty;
    public IFormFile Chunk { get; set; }
    public int CurrentChunk { get; set; }
    public int TotalChunks { get; set; }
}