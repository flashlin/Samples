using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System.IO;

namespace VimSharpApp.ApiHandlers;

public interface IUploadFileHandler
{
    Task<UploadFileResponse> Upload(UploadFileRequest req);
}

public class UploadFileRequest
{
    public required string FileName { get; set; }
    public required byte[] FileContent { get; set; }
    public required long Offset { get; set; }
}

public class UploadFileResponse
{
    public required string FileName { get; set; }
}

public class UploadFileHandler : IUploadFileHandler
{
    private const string UsersFileDir = @"D:\\demo\\UserFiles";

    public async Task<UploadFileResponse> Upload(UploadFileRequest req)
    {
        EnsureDirectoryExists();
        var filePath = GetFilePath(req.FileName);
        await WriteChunkToFile(filePath, req.FileContent, req.Offset);
        return new UploadFileResponse { FileName = req.FileName };
    }

    private void EnsureDirectoryExists()
    {
        if (!Directory.Exists(UsersFileDir))
        {
            Directory.CreateDirectory(UsersFileDir);
        }
    }

    private string GetFilePath(string fileName)
    {
        return Path.Combine(UsersFileDir, fileName);
    }

    private async Task WriteChunkToFile(string filePath, byte[] content, long offset)
    {
        await using var stream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
        stream.Seek(offset, SeekOrigin.Begin);
        await stream.WriteAsync(content, 0, content.Length);
        await stream.FlushAsync();
    }

    public static void MapEndpoints(WebApplication app)
    {
        app.MapPost("/api/UploadFile/Upload", (UploadFileRequest req,IUploadFileHandler handler) => handler.Upload(req));
    }
}