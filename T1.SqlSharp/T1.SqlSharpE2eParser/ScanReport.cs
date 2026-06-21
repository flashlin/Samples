namespace T1.SqlSharpE2eParser;

public sealed class ScanReport
{
    public required string SourcePath { get; init; }
    public required DateTimeOffset StartedAt { get; init; }
    public required DateTimeOffset FinishedAt { get; init; }
    public required ScanSummary Summary { get; init; }
    public required List<ErrorBucket> ErrorBuckets { get; init; }
    public required List<FileScanResult> Files { get; init; }

    public static ScanReport Create(string sourcePath, DateTimeOffset startedAt, TimeSpan elapsed, List<FileScanResult> files)
    {
        var finishedAt = startedAt.Add(elapsed);
        var errorBuckets = files
            .Where(x => x.FailedStatements > 0)
            .GroupBy(x => x.FirstErrorMessage)
            .Select(x => new ErrorBucket
            {
                Message = x.Key,
                Count = x.Count()
            })
            .OrderByDescending(x => x.Count)
            .ThenBy(x => x.Message, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return new ScanReport
        {
            SourcePath = sourcePath,
            StartedAt = startedAt,
            FinishedAt = finishedAt,
            Summary = ScanSummary.Create(files, elapsed),
            ErrorBuckets = errorBuckets,
            Files = files
        };
    }
}

public sealed class ScanSummary
{
    public required int TotalFiles { get; init; }
    public required int SucceededFiles { get; init; }
    public required int FailedFiles { get; init; }
    public required int TotalStatements { get; init; }
    public required int SucceededStatements { get; init; }
    public required int FailedStatements { get; init; }
    public required double ElapsedSeconds { get; init; }

    public static ScanSummary Create(List<FileScanResult> files, TimeSpan elapsed)
    {
        return new ScanSummary
        {
            TotalFiles = files.Count,
            SucceededFiles = files.Count(x => x.FailedStatements == 0),
            FailedFiles = files.Count(x => x.FailedStatements > 0),
            TotalStatements = files.Sum(x => x.StatementCount),
            SucceededStatements = files.Sum(x => x.SucceededStatements),
            FailedStatements = files.Sum(x => x.FailedStatements),
            ElapsedSeconds = elapsed.TotalSeconds
        };
    }
}

public sealed class ErrorBucket
{
    public required string Message { get; init; }
    public required int Count { get; init; }
}

public sealed class FileScanResult
{
    public required string FilePath { get; init; }
    public required int StatementCount { get; init; }
    public required int SucceededStatements { get; init; }
    public required int FailedStatements { get; init; }
    public required string FirstErrorMessage { get; init; }
    public required int? FirstErrorOffset { get; init; }
    public required double ElapsedMs { get; init; }
}
