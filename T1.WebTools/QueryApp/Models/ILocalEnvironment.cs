namespace QueryApp.Models;

public interface ILocalEnvironment
{
    int Port { get; set; }
    string MachineName { get; set; }
    string AppLocation { get; set; }
}

public class LocalEnvironment : ILocalEnvironment
{
    public string MachineName { get; set; } = string.Empty;
    public string AppLocation { get; set; } = string.Empty;
    public int Port { get; set; }
}