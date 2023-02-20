using System.Collections.Concurrent;
using T1.Standard.DesignPatterns;
using T1.WebTools.LocalQueryEx.Apis;

namespace T1.WebTools.LocalQueryEx;

/// <summary>
/// complete operation process steps
/// * user open front then download LocalQueryApp.exe and run LocalQueryApp.exe
/// * LocalQueryApp --> Echo every one sec until FrontUid is marked --> LocalQueryHost
/// * front --> GetAllLocalQueryApp Ports --> LocalQueryHost
/// * front --> Knock localhost: Ports and get AppUid --> LocalQueryApp
/// * front --> BindLocalQueryApp --> LocalQueryHost (mark FrontUid)
/// * front --> Send Api with token --> LocalQueryApp 
/// </summary>
public class LocalLocalQueryHostService : ILocalQueryHostService
{
    private readonly ConcurrentDictionary<string, LocalQueryEchoInfo> _localEchoInfos = new();

    public List<LocalQueryEchoInfo> GetUnbindLocalQueryApps()
    {
        return _localEchoInfos
            .Where(x => string.IsNullOrEmpty(x.Value.FrontUid))
            .Select(x => x.Value)
            .ToList();
    }

    public EchoResponse Echo(EchoRequest req)
    {
        var item = _localEchoInfos.AddOrUpdate(req.AppUid, key => new LocalQueryEchoInfo
        {
            AppUid = req.AppUid,
            Port = req.Port,
        }, (key, info) =>
        {
            info.LastActivityTime = DateTime.Now;
            return info;
        });
        return new EchoResponse
        {
            IsBinded = string.IsNullOrEmpty(item.FrontUid)
        };
    }

    public Either<LocalQueryEchoInfo, Exception> BindLocalQueryApp(BindLocalQueryAppRequest req)
    {
        if (!_localEchoInfos.TryGetValue(req.AppUid, out var item))
        {
            return new Either<LocalQueryEchoInfo, Exception>(new KeyNotFoundException($"AppUid: {req.AppUid}"));
        }

        lock (item)
        {
            if (!string.IsNullOrEmpty(item.FrontUid))
            {
                return new Either<LocalQueryEchoInfo, Exception>(new SynchronizationLockException($"Front Bind ERROR AppUid: {req.AppUid}"));
            }
            item.FrontUid = req.UniqueId;
        }

        return new Either<LocalQueryEchoInfo, Exception>(item);
    }
}