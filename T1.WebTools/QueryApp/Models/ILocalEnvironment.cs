namespace QueryApp.Models;

public interface ILocalEnvironment
{
    int Port { get; set; }
    string AppUid { get; set; }
    string AppLocation { get; set; }
    bool IsBinded { get; set; }
    DateTime LastActivityTime { get; set; }
    string UserUid { get; set; }
}