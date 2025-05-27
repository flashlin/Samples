using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace VimSharpApp.ApiHandlers;

public interface IUploadFileHandler
{
    Task<UploadFileResponse> Upload(UploadFileRequest req);
}

public class UploadFileRequest
{
    public required string FileName { get; set; }
    public string FileContent { get; set; }
}

public class UploadFileResponse
{
    public required string FileName { get; set; }
}

public class UploadFileHandler : IUploadFileHandler
{
    public Task<UploadFileResponse> Upload(UploadFileRequest req)
    {
        return Task.FromResult(new UploadFileResponse{
            FileName = req.FileName
        });       
    }

    public static void MapEndpoints(WebApplication app)
    {
        app.MapPost("/api/UploadFile/Upload", (UploadFileRequest req,IUploadFileHandler handler) => handler.Upload(req));
    }
}