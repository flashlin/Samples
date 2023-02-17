using T1.WebTools.LocalQueryEx;

namespace QueryApp.Models.Clients;

public interface ILocalQueryClient
{
    Task<EchoResponse> EchoAsync(ILocalEnvironment localEnvironment);
}