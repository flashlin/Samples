using Microsoft.Extensions.Hosting;
using QueryApp.Models.Clients;

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
            if (!_localEnvironment.IsBinded)
            {
                var resp = await _localQueryHostClient.EchoAsync(_localEnvironment);
                _localEnvironment.IsBinded = resp.IsBinded;
            }
            else
            {
                if (_localEnvironment.LastActivityTime.AddSeconds(10) < DateTime.Now)
                {
                    _localEnvironment.IsBinded = false;
                }
            }
            await Task.Delay(1000, stoppingToken);
        }
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