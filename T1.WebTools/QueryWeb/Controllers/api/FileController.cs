using Microsoft.AspNetCore.Mvc;

namespace QueryWeb.Controllers.api;

//[ApiController]
//[Route("api/[controller]/[action]")]
public class FileController : ControllerBase
{
    [HttpPost]
    public IActionResult Upload([FromForm] UploadFilesRequest req)
    {
        return Ok();
    }
}

public class UploadFilesRequest
{
    public string FileName { get; set; } = string.Empty;
    public byte[] Chunk { get; set; } = Array.Empty<byte>();
    public int CurrentChunk { get; set; }
    public int TotalChunks { get; set; }
}