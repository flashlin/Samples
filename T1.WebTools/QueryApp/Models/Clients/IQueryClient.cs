namespace QueryApp.Models.Clients;

public interface IQueryClient
{
    Task EchoAsync(ILocalEnvironment localEnvironment);
    void KnockAsync(KnockRequest req);
}