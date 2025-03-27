namespace VimSharpApp.ApiHandlers;

public class JobInfo
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Method { get; set; } = string.Empty;
    public List<string> Parameters { get; set; } = [];
}
