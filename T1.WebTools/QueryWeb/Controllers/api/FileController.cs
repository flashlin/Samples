using Microsoft.AspNetCore.Mvc;

namespace QueryWeb.Controllers.api;

[ApiController]
[Route("api/[controller]/[action]")]
public class FileController : ControllerBase
{
    [HttpPost]
    public string SayHello([FromBody] string name)
    {
        return $"Hello {name}";
    }
    
    [HttpPost]
    public IActionResult Upload([FromForm] UploadFilesRequest req)
    {
        using var file = new FileStream("d:\\demo\\222.txt", FileMode.OpenOrCreate);
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