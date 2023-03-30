using QueryKits.Services;

namespace QueryWeb.Models;

public class QueryEnvironment : IQueryEnvironment
{
    public QueryEnvironment(IWebHostEnvironment webHostEnv)
    {
        UploadPath = Path.Combine(webHostEnv.WebRootPath, "Upload");
    }
    public string UploadPath { get; set; }
}