namespace QueryApp.Models;

public class LocalEnvironment : ILocalEnvironment
{
    public string MachineName { get; set; } = string.Empty;
    public string AppLocation { get; set; } = string.Empty;
    public int Port { get; set; }
}