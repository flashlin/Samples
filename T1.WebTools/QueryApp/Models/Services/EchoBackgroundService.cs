using Microsoft.Extensions.Hosting;

namespace QueryApp.Models.Services;

public class EchoBackgroundService : IHostedService, IDisposable
{
    private Timer? _timer;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        Console.WriteLine("Starting background service...");
        _timer = new Timer(DoWork!, null, TimeSpan.Zero, TimeSpan.FromSeconds(1));
        return Task.CompletedTask;
    }

    private void DoWork(object state)
    {
        Console.WriteLine("Hello");
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        Console.WriteLine("Stopping background service...");
        _timer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        _timer?.Dispose();
    }
}