using System.Text;
using System.Text.Json;

namespace T1.SqlSharpE2eParser;

public static class ScanReportWriter
{
    public const string JsonFileName = "report.json";
    public const string CsvFileName = "report.csv";
    public const string SummaryFileName = "summary.json";
    public const string ErrorCsvFileName = "error.csv";
    public const string ErrorSummaryCsvFileName = "error-summary.csv";
    public const string CsvHeader = "FilePath,StatementCount,SucceededStatements,FailedStatements,FirstErrorOffset,FirstErrorMessage,ElapsedMs";

    public static void Write(ScanReport report, string outputPath)
    {
        Directory.CreateDirectory(outputPath);
        WriteJson(report, Path.Combine(outputPath, JsonFileName));
        WriteCsv(report.Files, Path.Combine(outputPath, CsvFileName));
    }

    private static void WriteJson(ScanReport report, string filePath)
    {
        var json = JsonSerializer.Serialize(report, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(filePath, json, Encoding.UTF8);
    }

    private static void WriteCsv(IEnumerable<FileScanResult> files, string filePath)
    {
        var csv = new StringBuilder();
        csv.AppendLine(CsvHeader);
        foreach (var file in files)
        {
            csv.AppendLine(ToCsvLine(file));
        }
        File.WriteAllText(filePath, csv.ToString(), Encoding.UTF8);
    }

    public static string ToCsvLine(FileScanResult file)
    {
        return string.Join(",", [
            EscapeCsv(file.FilePath),
            file.StatementCount.ToString(),
            file.SucceededStatements.ToString(),
            file.FailedStatements.ToString(),
            file.FirstErrorOffset?.ToString() ?? string.Empty,
            EscapeCsv(file.FirstErrorMessage),
            file.ElapsedMs.ToString("F3")
        ]);
    }

    public static string EscapeCsv(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return $"\"{value.Replace("\"", "\"\"")}\"";
    }
}
