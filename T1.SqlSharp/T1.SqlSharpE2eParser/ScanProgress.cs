namespace T1.SqlSharpE2eParser;

public sealed class ScanProgress
{
    private readonly int _totalFiles;
    private readonly object _syncRoot = new();
    private int _processedFiles;
    private int _succeededFiles;
    private int _failedFiles;

    public ScanProgress(int totalFiles)
    {
        _totalFiles = totalFiles;
    }

    public ScanProgressSnapshot Record(FileScanResult result)
    {
        lock (_syncRoot)
        {
            _processedFiles++;
            if (result.FailedStatements == 0)
            {
                _succeededFiles++;
            }
            else
            {
                _failedFiles++;
            }

            if (ShouldPrintProgress())
            {
                Console.WriteLine($"Processed {_processedFiles}/{_totalFiles} | OK {_succeededFiles} | FAIL {_failedFiles}");
            }

            return new ScanProgressSnapshot
            {
                TotalFiles = _totalFiles,
                ProcessedFiles = _processedFiles,
                SucceededFiles = _succeededFiles,
                FailedFiles = _failedFiles,
                ShouldWriteSummary = ShouldPrintProgress()
            };
        }
    }

    private bool ShouldPrintProgress()
    {
        return _processedFiles == _totalFiles || _processedFiles % 100 == 0;
    }
}

public sealed class ScanProgressSnapshot
{
    public required int TotalFiles { get; init; }
    public required int ProcessedFiles { get; init; }
    public required int SucceededFiles { get; init; }
    public required int FailedFiles { get; init; }
    public required bool ShouldWriteSummary { get; init; }
}
