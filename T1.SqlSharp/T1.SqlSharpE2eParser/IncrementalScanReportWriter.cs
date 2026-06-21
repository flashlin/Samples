using System.Text;
using System.Text.Json;

namespace T1.SqlSharpE2eParser;

public sealed class IncrementalScanReportWriter
{
    private readonly string _outputPath;
    private readonly string _csvPath;
    private readonly string _summaryPath;
    private readonly object _syncRoot = new();

    public IncrementalScanReportWriter(string outputPath)
    {
        _outputPath = outputPath;
        _csvPath = Path.Combine(outputPath, ScanReportWriter.CsvFileName);
        _summaryPath = Path.Combine(outputPath, ScanReportWriter.SummaryFileName);
        Directory.CreateDirectory(outputPath);
        File.WriteAllText(_csvPath, ScanReportWriter.CsvHeader + Environment.NewLine, Encoding.UTF8);
    }

    public void AppendFile(FileScanResult result)
    {
        lock (_syncRoot)
        {
            File.AppendAllText(_csvPath, ScanReportWriter.ToCsvLine(result) + Environment.NewLine, Encoding.UTF8);
        }
    }

    public void WriteSummary(string sourcePath, DateTimeOffset startedAt, ScanProgressSnapshot progress)
    {
        lock (_syncRoot)
        {
            WriteSummaryFile(new
            {
                SourcePath = sourcePath,
                StartedAt = startedAt,
                LastUpdatedAt = DateTimeOffset.UtcNow,
                progress.TotalFiles,
                progress.ProcessedFiles,
                progress.SucceededFiles,
                progress.FailedFiles,
                IsCompleted = progress.ProcessedFiles == progress.TotalFiles
            });
        }
    }

    public void WriteSummary(ScanReport report)
    {
        lock (_syncRoot)
        {
            WriteSummaryFile(new
            {
                report.SourcePath,
                report.StartedAt,
                LastUpdatedAt = report.FinishedAt,
                report.Summary.TotalFiles,
                ProcessedFiles = report.Summary.TotalFiles,
                report.Summary.SucceededFiles,
                report.Summary.FailedFiles,
                IsCompleted = true
            });
        }
    }

    private void WriteSummaryFile<T>(T summary)
    {
        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(_summaryPath, json, Encoding.UTF8);
    }
}
