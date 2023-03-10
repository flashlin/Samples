namespace QueryKits.Services;

public class LocalEnvironment : ILocalEnvironment
{
    public string AppUid { get; set; } = string.Empty;
    public string AppLocation { get; set; } = string.Empty;
    public bool IsBinded { get; set; }
    public DateTime LastActivityTime { get; set; }
    public string UserUid { get; set; } = string.Empty;
    public int Port { get; set; }
    public string AppVersion { get; set; } = string.Empty;
}