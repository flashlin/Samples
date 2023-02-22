using Microsoft.Extensions.Hosting;
using QueryApp.Models.Clients;
using T1.WebTools.LocalQueryEx;

namespace QueryApp.Models.Services;

public class EchoBackgroundService : BackgroundService
{
    private readonly ILocalEnvironment _localEnvironment;
    private readonly ILocalQueryHostClient _localQueryHostClient;

    public EchoBackgroundService(ILocalEnvironment localEnvironment, ILocalQueryHostClient localQueryHostClient)
    {
        _localQueryHostClient = localQueryHostClient;
        _localEnvironment = localEnvironment;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(1000, stoppingToken);
            if (!_localEnvironment.IsBinded)
            {
                try
                {
                    await _localQueryHostClient.EchoAsync(_localEnvironment);
                }
                catch
                {
                    continue;
                }
                continue;
            }
            
            if (_localEnvironment.LastActivityTime.AddSeconds(10) < DateTime.Now)
            {
                _localEnvironment.UserUid = string.Empty;
                _localEnvironment.IsBinded = false;
            }
        }
    }
}