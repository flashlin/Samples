using T1.Standard.DesignPatterns;
using T1.WebTools.LocalQueryEx.Apis;

namespace T1.WebTools.LocalQueryEx;

public interface ILocalQueryHostService
{
    EchoResponse Echo(EchoRequest req);
    BindLocalQueryAppResponse BindLocalQueryApp(BindLocalQueryAppRequest req);
    List<LocalQueryEchoInfo> GetUnbindLocalQueryApps();
}