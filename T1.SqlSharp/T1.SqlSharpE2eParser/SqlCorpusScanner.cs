using System.Collections.Concurrent;
using System.Diagnostics;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpE2eParser;

public sealed class SqlCorpusScanner
{
    public ScanReport Scan(string sourcePath, string outputPath)
    {
        var files = Directory
            .EnumerateFiles(sourcePath, "*.sql", SearchOption.AllDirectories)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var startedAt = DateTimeOffset.UtcNow;
        var stopwatch = Stopwatch.StartNew();
        var results = new ConcurrentBag<FileScanResult>();
        var progress = new ScanProgress(files.Length);
        var incrementalWriter = new IncrementalScanReportWriter(outputPath);
        var fileQueue = new ConcurrentQueue<string>(files);
        var workers = Enumerable
            .Range(0, Environment.ProcessorCount)
            .Select(_ => Task.Run(() => ScanFilesWithWorker(sourcePath, startedAt, fileQueue, results, progress, incrementalWriter)))
            .ToArray();

        Task.WaitAll(workers);

        stopwatch.Stop();
        var orderedResults = results
            .OrderBy(x => x.FilePath, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var report = ScanReport.Create(sourcePath, startedAt, stopwatch.Elapsed, orderedResults);
        incrementalWriter.WriteSummary(report);
        return report;
    }

    public static FileScanResult ScanFileInProcess(string sourcePath, string filePath)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            var text = File.ReadAllText(filePath);
            var statementResults = new SqlParser(text).ExtractStatementResults().ToList();
            stopwatch.Stop();

            var error = statementResults.FirstOrDefault(x => x.HasError);
            var succeededStatements = statementResults.Count(x => !x.HasError);
            var failedStatements = error == null ? 0 : 1;

            return new FileScanResult
            {
                FilePath = Path.GetRelativePath(sourcePath, filePath),
                StatementCount = statementResults.Count,
                SucceededStatements = succeededStatements,
                FailedStatements = failedStatements,
                FirstErrorMessage = error?.Error.Message ?? string.Empty,
                FirstErrorOffset = error?.Error.Offset,
                ElapsedMs = stopwatch.Elapsed.TotalMilliseconds
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            return new FileScanResult
            {
                FilePath = Path.GetRelativePath(sourcePath, filePath),
                StatementCount = 1,
                SucceededStatements = 0,
                FailedStatements = 1,
                FirstErrorMessage = ex.Message,
                FirstErrorOffset = null,
                ElapsedMs = stopwatch.Elapsed.TotalMilliseconds
            };
        }
    }

    private static void ScanFilesWithWorker(
        string sourcePath,
        DateTimeOffset startedAt,
        ConcurrentQueue<string> fileQueue,
        ConcurrentBag<FileScanResult> results,
        ScanProgress progress,
        IncrementalScanReportWriter incrementalWriter)
    {
        using var worker = new SqlFileWorker(sourcePath);
        while (fileQueue.TryDequeue(out var filePath))
        {
            var result = worker.Scan(filePath);
            results.Add(result);
            var progressSnapshot = progress.Record(result);
            incrementalWriter.AppendFile(result);
            if (progressSnapshot.ShouldWriteSummary)
            {
                incrementalWriter.WriteSummary(sourcePath, startedAt, progressSnapshot);
            }
        }
    }
}
