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

public enum ExcelDataType
{
    String,
    Number
}

public class ExcelColumn
{
    public string Name { get; set; } = string.Empty;
    public ExcelDataType DataType { get; set; }
    public int CellIndex { get; set; }
}

public class ExcelSheet
{
    public string Name { get; set; } = string.Empty;
    public List<ExcelColumn> Header { get; set; } = new();
    public List<Dictionary<string, string>> Rows { get; set; } = new();
}