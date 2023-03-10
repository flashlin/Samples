using System.Net;
using System.Net.Sockets;
using System.Reflection;

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

    public static LocalEnvironment Load()
    {
        var entryAssembly = Assembly.GetEntryAssembly()!;
        var appUid = Guid.NewGuid().ToString();
        var appLocation = AppContext.BaseDirectory;
        var appVersion = entryAssembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!
            .InformationalVersion;
        var port = FindAvailablePort();
        return new LocalEnvironment
        {
            AppUid = appUid,
            AppLocation = appLocation,
            IsBinded = false,
            LastActivityTime = DateTime.Now,
            UserUid = null,
            Port = 0,
            AppVersion = appVersion 
        };
    }
    
    
    static int FindAvailablePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint) listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}