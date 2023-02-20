using T1.Standard.DesignPatterns;

namespace T1.WebTools.LocalQueryEx;

public interface ILocalQueryHostService
{
    EchoResponse Echo(EchoRequest req);
    List<LocalQueryEchoInfo> GetUnbindLocalQueryApps();
    void UnEcho(UnEchoRequest req);
}