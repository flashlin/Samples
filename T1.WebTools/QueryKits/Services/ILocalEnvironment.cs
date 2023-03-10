namespace QueryKits.Services;

public interface ILocalEnvironment
{
    int Port { get; set; }
    string AppUid { get; set; }
    bool IsBinded { get; set; }
    DateTime LastActivityTime { get; set; }
    string UserUid { get; set; }
    string AppLocation { get; set; }
    string AppVersion { get; set; }
}