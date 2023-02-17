namespace QueryApp.Models;

public interface ILocalEnvironment
{
    int Port { get; set; }
    string MachineName { get; set; }
    string AppLocation { get; set; }
}