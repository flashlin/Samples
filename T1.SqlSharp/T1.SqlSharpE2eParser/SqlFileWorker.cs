using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

namespace T1.SqlSharpE2eParser;

public sealed class SqlFileWorker : IDisposable
{
    private const int WorkerTimeoutMs = 30000;

    private readonly string _sourcePath;
    private readonly object _syncRoot = new();
    private Process? _process;
    private string _firstErrorLine = string.Empty;

    public SqlFileWorker(string sourcePath)
    {
        _sourcePath = sourcePath;
        Start();
    }

    public FileScanResult Scan(string filePath)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            var process = EnsureProcess();
            process.StandardInput.WriteLine(filePath);
            process.StandardInput.Flush();

            var outputTask = process.StandardOutput.ReadLineAsync();
            if (!outputTask.Wait(WorkerTimeoutMs))
            {
                Restart();
                stopwatch.Stop();
                return CreateWorkerFailure(filePath, "Worker process timed out", stopwatch.Elapsed);
            }

            var output = outputTask.Result;
            stopwatch.Stop();
            if (string.IsNullOrWhiteSpace(output))
            {
                var errorMessage = ReadWorkerError();
                Restart();
                return CreateWorkerFailure(filePath, errorMessage, stopwatch.Elapsed);
            }

            return JsonSerializer.Deserialize<FileScanResult>(output)
                   ?? CreateWorkerFailure(filePath, "Worker process returned empty result", stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Restart();
            return CreateWorkerFailure(filePath, ex.Message, stopwatch.Elapsed);
        }
    }

    public void Dispose()
    {
        Stop();
    }

    private Process EnsureProcess()
    {
        if (_process is { HasExited: false })
        {
            return _process;
        }

        Start();
        return _process ?? throw new InvalidOperationException("Worker process is not available");
    }

    private void Start()
    {
        _process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "dotnet",
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            }
        };

        _process.StartInfo.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        _process.StartInfo.ArgumentList.Add("--worker");
        _process.StartInfo.ArgumentList.Add(_sourcePath);
        _process.ErrorDataReceived += OnErrorDataReceived;
        _process.Start();
        _process.BeginErrorReadLine();
    }

    private void Restart()
    {
        Stop();
        Start();
    }

    private void Stop()
    {
        if (_process == null)
        {
            return;
        }

        try
        {
            if (!_process.HasExited)
            {
                _process.StandardInput.Close();
                if (!_process.WaitForExit(1000))
                {
                    _process.Kill(true);
                }
            }
        }
        catch
        {
        }
        finally
        {
            _process.Dispose();
            _process = null;
            _firstErrorLine = string.Empty;
        }
    }

    private FileScanResult CreateWorkerFailure(string filePath, string message, TimeSpan elapsed)
    {
        return new FileScanResult
        {
            FilePath = Path.GetRelativePath(_sourcePath, filePath),
            StatementCount = 1,
            SucceededStatements = 0,
            FailedStatements = 1,
            FirstErrorMessage = message,
            FirstErrorOffset = null,
            ElapsedMs = elapsed.TotalMilliseconds
        };
    }

    private string ReadWorkerError()
    {
        lock (_syncRoot)
        {
            if (string.IsNullOrWhiteSpace(_firstErrorLine))
            {
                return "Worker process failed";
            }

            return $"Worker process failed: {_firstErrorLine}";
        }
    }

    private void OnErrorDataReceived(object sender, DataReceivedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(e.Data))
        {
            return;
        }

        lock (_syncRoot)
        {
            if (string.IsNullOrWhiteSpace(_firstErrorLine))
            {
                _firstErrorLine = e.Data;
            }
        }
    }
}
