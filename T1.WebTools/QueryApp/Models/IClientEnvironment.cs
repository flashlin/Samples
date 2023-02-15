namespace QueryApp.Models;

public interface IClientEnvironment
{
    int Port { get; set; }
}

public class ClientEnvironment : IClientEnvironment
{
    public int Port { get; set; }
}