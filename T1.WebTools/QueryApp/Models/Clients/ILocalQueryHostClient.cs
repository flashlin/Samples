using T1.WebTools.LocalQueryEx;

namespace QueryApp.Models.Clients;

public interface ILocalQueryHostClient
{
    Task<EchoResponse> EchoAsync(ILocalEnvironment localEnvironment);
}