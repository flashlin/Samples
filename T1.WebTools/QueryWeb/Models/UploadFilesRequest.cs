namespace QueryWeb.Models;

public class UploadFilesRequest
{
    public string FileName { get; set; } = string.Empty;
    public IFormFile Chunk { get; set; }
    public int CurrentChunk { get; set; }
    public int TotalChunks { get; set; }
}