namespace QueryApp.Models;

public class KnockResponse
{
    public bool IsSuccess { get; set; }
    public string AppVersion { get; set; } = string.Empty;
}