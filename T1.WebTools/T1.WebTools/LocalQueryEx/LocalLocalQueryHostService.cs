using System.Collections.Concurrent;
using T1.Standard.DesignPatterns;

namespace T1.WebTools.LocalQueryEx;

/// <summary>
/// complete operation process steps
/// * user open front then download LocalQueryApp.exe and run LocalQueryApp.exe
/// * LocalQueryApp --> Echo every one sec until FrontUid is marked --> LocalQueryHost
/// * front --> GetUnbindLocalQueryApps --> LocalQueryHost
/// * front --> Knock                   --> LocalQueryApp
/// * front --> BindLocalQueryApp       --> LocalQueryHost
/// * front --> Send Api with token     --> LocalQueryApp 
/// </summary>
public class LocalLocalQueryHostService : ILocalQueryHostService
{
    private readonly ConcurrentDictionary<string, LocalQueryEchoInfo> _localEchoInfos = new();

    public List<LocalQueryEchoInfo> GetUnbindLocalQueryApps()
    {
        return _localEchoInfos
            .Where(x => DateTime.Now < x.Value.LastActivityTime.AddSeconds(5))
            .Select(x => x.Value)
            .ToList();
    }

    public EchoResponse Echo(EchoRequest req)
    {
        _localEchoInfos.AddOrUpdate(req.AppUid, key => new LocalQueryEchoInfo
        {
            AppUid = req.AppUid,
            Port = req.Port,
            LastActivityTime = DateTime.Now,
        }, (key, info) =>
        {
            info.Port = req.Port;
            info.LastActivityTime = DateTime.Now;
            return info;
        });
        return new EchoResponse();
    }
}