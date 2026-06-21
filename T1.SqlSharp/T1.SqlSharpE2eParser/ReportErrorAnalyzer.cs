using System.Text;
using System.Text.RegularExpressions;

namespace T1.SqlSharpE2eParser;

public sealed class ReportErrorAnalyzer
{
    private const int PreviewLength = 240;
    private const int ContextRadius = 160;
    private const int ClassificationContextRadius = 1400;

    public ReportErrorAnalysisResult Analyze(string sourcePath, string outputPath)
    {
        var reportPath = Path.Combine(outputPath, ScanReportWriter.CsvFileName);
        var errorPath = Path.Combine(outputPath, ScanReportWriter.ErrorCsvFileName);
        var summaryPath = Path.Combine(outputPath, ScanReportWriter.ErrorSummaryCsvFileName);

        var errors = ReadReport(reportPath)
            .Where(x => x.FailedStatements > 0)
            .Select(x => CreateErrorRow(sourcePath, x))
            .ToList();

        WriteErrorCsv(errorPath, errors);
        WriteSummaryCsv(summaryPath, errors);

        return new ReportErrorAnalysisResult
        {
            ErrorCsvPath = errorPath,
            ErrorSummaryCsvPath = summaryPath
        };
    }

    private static IEnumerable<ReportRow> ReadReport(string reportPath)
    {
        foreach (var line in File.ReadLines(reportPath).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var fields = ParseCsvLine(line);
            if (fields.Count < 7)
            {
                continue;
            }

            yield return new ReportRow
            {
                FilePath = fields[0],
                StatementCount = ParseInt(fields[1]),
                SucceededStatements = ParseInt(fields[2]),
                FailedStatements = ParseInt(fields[3]),
                FirstErrorOffset = ParseNullableInt(fields[4]),
                FirstErrorMessage = fields[5],
                ElapsedMs = fields[6]
            };
        }
    }

    private static ErrorReportRow CreateErrorRow(string sourcePath, ReportRow row)
    {
        var fullPath = Path.Combine(sourcePath, row.FilePath);
        if (!File.Exists(fullPath))
        {
            return CreateMissingFileRow(row);
        }

        var sql = File.ReadAllText(fullPath);
        var offset = ClampOffset(row.FirstErrorOffset ?? 0, sql);
        var errorLine = GetLineNumber(sql, offset);
        var statementStart = FindStatementStart(sql, offset);
        var statementStartLine = GetLineNumber(sql, statementStart);
        var statementPreview = NormalizePreview(ReadPreview(sql, statementStart, PreviewLength));
        var contextPreview = NormalizePreview(ReadContext(sql, offset, ContextRadius));
        var classificationContext = NormalizePreview(ReadContext(sql, offset, ClassificationContextRadius));
        var statementKind = DetectStatementKind(sql, offset, statementStart);
        var category = Classify(row.FirstErrorMessage, statementKind, statementPreview, contextPreview, classificationContext);

        return new ErrorReportRow
        {
            Category = category,
            ErrorMessage = row.FirstErrorMessage,
            FilePath = row.FilePath,
            FirstErrorOffset = row.FirstErrorOffset?.ToString() ?? string.Empty,
            ErrorLine = errorLine,
            StatementStartLine = statementStartLine,
            StatementKind = statementKind,
            StatementPreview = statementPreview,
            ContextPreview = contextPreview,
            SuggestedTestName = CreateSuggestedTestName(category, statementKind)
        };
    }

    private static ErrorReportRow CreateMissingFileRow(ReportRow row)
    {
        return new ErrorReportRow
        {
            Category = "Missing SQL file",
            ErrorMessage = row.FirstErrorMessage,
            FilePath = row.FilePath,
            FirstErrorOffset = row.FirstErrorOffset?.ToString() ?? string.Empty,
            ErrorLine = 0,
            StatementStartLine = 0,
            StatementKind = string.Empty,
            StatementPreview = string.Empty,
            ContextPreview = string.Empty,
            SuggestedTestName = "Missing_sql_file"
        };
    }

    private static void WriteErrorCsv(string path, IReadOnlyList<ErrorReportRow> errors)
    {
        var csv = new StringBuilder();
        csv.AppendLine("Category,ErrorMessage,FilePath,FirstErrorOffset,ErrorLine,StatementStartLine,StatementKind,StatementPreview,ContextPreview,SuggestedTestName");
        foreach (var error in errors)
        {
            csv.AppendLine(string.Join(",", [
                ScanReportWriter.EscapeCsv(error.Category),
                ScanReportWriter.EscapeCsv(error.ErrorMessage),
                ScanReportWriter.EscapeCsv(error.FilePath),
                ScanReportWriter.EscapeCsv(error.FirstErrorOffset),
                error.ErrorLine.ToString(),
                error.StatementStartLine.ToString(),
                ScanReportWriter.EscapeCsv(error.StatementKind),
                ScanReportWriter.EscapeCsv(error.StatementPreview),
                ScanReportWriter.EscapeCsv(error.ContextPreview),
                ScanReportWriter.EscapeCsv(error.SuggestedTestName)
            ]));
        }
        File.WriteAllText(path, csv.ToString(), Encoding.UTF8);
    }

    private static void WriteSummaryCsv(string path, IReadOnlyList<ErrorReportRow> errors)
    {
        var csv = new StringBuilder();
        csv.AppendLine("Category,ErrorMessage,StatementKind,Count,RepresentativeFile,RepresentativeLine,RepresentativePreview,SuggestedTestName");
        var groups = errors
            .GroupBy(x => new { x.Category, x.ErrorMessage, x.StatementKind })
            .OrderByDescending(x => x.Count())
            .ThenBy(x => x.Key.Category, StringComparer.OrdinalIgnoreCase)
            .ThenBy(x => x.Key.StatementKind, StringComparer.OrdinalIgnoreCase);

        foreach (var group in groups)
        {
            var representative = group.First();
            csv.AppendLine(string.Join(",", [
                ScanReportWriter.EscapeCsv(group.Key.Category),
                ScanReportWriter.EscapeCsv(group.Key.ErrorMessage),
                ScanReportWriter.EscapeCsv(group.Key.StatementKind),
                group.Count().ToString(),
                ScanReportWriter.EscapeCsv(representative.FilePath),
                representative.StatementStartLine.ToString(),
                ScanReportWriter.EscapeCsv(representative.StatementPreview),
                ScanReportWriter.EscapeCsv(representative.SuggestedTestName)
            ]));
        }
        File.WriteAllText(path, csv.ToString(), Encoding.UTF8);
    }

    private static string Classify(string errorMessage, string statementKind, string statementPreview, string contextPreview, string classificationContext)
    {
        var text = $"{statementPreview} {contextPreview} {classificationContext}";
        var normalized = text.ToUpperInvariant();

        if (errorMessage.StartsWith("Worker process failed", StringComparison.OrdinalIgnoreCase))
        {
            return "Parser process crash";
        }

        if (errorMessage == "Result is null")
        {
            return "Parser returned null result";
        }

        if (normalized.Contains("IDENTITY_INSERT"))
        {
            return "SET IDENTITY_INSERT script option";
        }

        if (IsInsertValuesContinuation(statementKind, statementPreview, contextPreview, classificationContext))
        {
            return "INSERT multi-row VALUES continuation";
        }

        if (normalized.Contains("GRANT EXEC") || normalized.Contains("GRANT SELECT") || normalized.Contains("GRANT INSERT"))
        {
            return string.IsNullOrWhiteSpace(statementPreview)
                ? "Permission statement trailing batch separator"
                : "Permission statement variant";
        }

        if (normalized.Contains("CREATE PARTITION SCHEME"))
        {
            return "CREATE PARTITION SCHEME variant";
        }

        if (normalized.Contains("CREATE SYNONYM"))
        {
            return "CREATE SYNONYM variant";
        }

        if (Regex.IsMatch(normalized, @"\bINSERT\s+\[[^\]]+\]") || Regex.IsMatch(normalized, @"\bINSERT\s+DBO\."))
        {
            return "INSERT data script without INTO";
        }

        if (normalized.Contains("CAST(0X"))
        {
            return "Binary literal CAST value";
        }

        if (normalized.StartsWith("IF ") || normalized.StartsWith("IF("))
        {
            return "IF conditional script";
        }

        if (normalized.StartsWith("EXEC ") || normalized.StartsWith("EXECUTE "))
        {
            return "EXEC statement variant";
        }

        if (normalized.StartsWith("GRANT ") || normalized.StartsWith("DENY ") || normalized.StartsWith("REVOKE "))
        {
            return "Permission statement variant";
        }

        if (normalized.StartsWith("CREATE PROCEDURE") || normalized.StartsWith("CREATE PROC"))
        {
            return "CREATE PROCEDURE variant";
        }

        if (normalized.StartsWith("CREATE FUNCTION"))
        {
            return "CREATE FUNCTION variant";
        }

        if (normalized.StartsWith("CREATE TABLE"))
        {
            return "CREATE TABLE variant";
        }

        if (normalized.StartsWith("ALTER TABLE"))
        {
            return "ALTER TABLE variant";
        }

        if (normalized.StartsWith("DECLARE "))
        {
            return "DECLARE variant";
        }

        if (normalized.StartsWith("MERGE "))
        {
            return "MERGE variant";
        }

        if (!string.IsNullOrWhiteSpace(statementKind))
        {
            return $"Unsupported or partial {statementKind}";
        }

        return "Unclassified parser failure";
    }

    private static bool IsInsertValuesContinuation(string statementKind, string statementPreview, string contextPreview, string classificationContext)
    {
        if (statementPreview.TrimStart().StartsWith('('))
        {
            return true;
        }

        if (!string.IsNullOrWhiteSpace(statementKind) && !statementKind.StartsWith("INSERT", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var normalizedContext = classificationContext.ToUpperInvariant();
        if (!Regex.IsMatch(normalizedContext, @"\bVALUES\b"))
        {
            return false;
        }

        var normalizedErrorContext = contextPreview.ToUpperInvariant();
        return normalizedErrorContext.Contains("),") || normalizedErrorContext.Contains(");");
    }

    private static string DetectStatementKind(string sql, int offset, int statementStart)
    {
        var start = SkipWhitespaceAndComments(sql, statementStart);
        var statementKind = ReadStatementKind(sql, start);
        if (!string.IsNullOrWhiteSpace(statementKind))
        {
            return statementKind;
        }

        start = SkipWhitespaceAndComments(sql, Math.Min(offset, sql.Length));
        return ReadStatementKind(sql, start);
    }

    private static string ReadStatementKind(string sql, int start)
    {
        if (start >= sql.Length)
        {
            return string.Empty;
        }

        var preview = ReadPreview(sql, start, 80).TrimStart();
        var match = Regex.Match(preview, @"^([A-Za-z]+)(?:\s+([A-Za-z]+))?");
        if (!match.Success)
        {
            return string.Empty;
        }

        var first = match.Groups[1].Value.ToUpperInvariant();
        var second = match.Groups[2].Success ? match.Groups[2].Value.ToUpperInvariant() : string.Empty;
        return string.IsNullOrWhiteSpace(second) ? first : $"{first} {second}";
    }

    private static int FindStatementStart(string sql, int offset)
    {
        var cursor = ClampOffset(offset, sql);
        var lineStart = sql.LastIndexOf('\n', Math.Max(0, cursor - 1));
        var statementStart = lineStart < 0 ? 0 : lineStart + 1;

        while (statementStart > 0)
        {
            var previousLineEnd = statementStart - 1;
            var previousLineStart = sql.LastIndexOf('\n', Math.Max(0, previousLineEnd - 1));
            previousLineStart = previousLineStart < 0 ? 0 : previousLineStart + 1;
            var previousLineLength = Math.Max(0, previousLineEnd - previousLineStart);
            var previousLine = sql.Substring(previousLineStart, previousLineLength).Trim();

            if (previousLine.Length == 0 || previousLine.Equals("GO", StringComparison.OrdinalIgnoreCase) || previousLine.EndsWith(';'))
            {
                break;
            }

            statementStart = previousLineStart;
        }

        return SkipWhitespaceAndComments(sql, statementStart);
    }

    private static int SkipWhitespaceAndComments(string sql, int offset)
    {
        var cursor = ClampOffset(offset, sql);
        while (cursor < sql.Length)
        {
            while (cursor < sql.Length && char.IsWhiteSpace(sql[cursor]))
            {
                cursor++;
            }

            if (cursor + 1 < sql.Length && sql[cursor] == '-' && sql[cursor + 1] == '-')
            {
                cursor = sql.IndexOf('\n', cursor);
                if (cursor < 0)
                {
                    return sql.Length;
                }
                continue;
            }

            if (cursor + 1 < sql.Length && sql[cursor] == '/' && sql[cursor + 1] == '*')
            {
                var end = sql.IndexOf("*/", cursor + 2, StringComparison.Ordinal);
                cursor = end < 0 ? sql.Length : end + 2;
                continue;
            }

            break;
        }

        return cursor;
    }

    private static int GetLineNumber(string sql, int offset)
    {
        offset = ClampOffset(offset, sql);
        var line = 1;
        for (var i = 0; i < offset; i++)
        {
            if (sql[i] == '\n')
            {
                line++;
            }
        }
        return line;
    }

    private static string ReadContext(string sql, int offset, int radius)
    {
        offset = ClampOffset(offset, sql);
        var start = Math.Max(0, offset - radius);
        var length = Math.Min(sql.Length - start, radius * 2);
        return sql.Substring(start, length);
    }

    private static string ReadPreview(string sql, int offset, int length)
    {
        offset = ClampOffset(offset, sql);
        return sql.Substring(offset, Math.Min(length, sql.Length - offset));
    }

    private static string NormalizePreview(string text)
    {
        return Regex.Replace(text, @"\s+", " ").Trim();
    }

    private static int ClampOffset(int offset, string sql)
    {
        return Math.Clamp(offset, 0, sql.Length);
    }

    private static int ParseInt(string text)
    {
        return int.TryParse(text, out var value) ? value : 0;
    }

    private static int? ParseNullableInt(string text)
    {
        return int.TryParse(text, out var value) ? value : null;
    }

    private static List<string> ParseCsvLine(string line)
    {
        if (line.Length > 0 && line[0] == '\ufeff')
        {
            line = line[1..];
        }

        var fields = new List<string>();
        var current = new StringBuilder();
        var inQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (c == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    current.Append('"');
                    i++;
                    continue;
                }

                inQuotes = !inQuotes;
                continue;
            }

            if (c == ',' && !inQuotes)
            {
                fields.Add(current.ToString());
                current.Clear();
                continue;
            }

            current.Append(c);
        }

        fields.Add(current.ToString());
        return fields;
    }

    private static string CreateSuggestedTestName(string category, string statementKind)
    {
        var text = string.IsNullOrWhiteSpace(statementKind)
            ? category
            : $"{category}_{statementKind}";

        return Regex.Replace(text, @"[^A-Za-z0-9]+", "_").Trim('_');
    }
}

public sealed class ReportErrorAnalysisResult
{
    public required string ErrorCsvPath { get; init; }
    public required string ErrorSummaryCsvPath { get; init; }
}

public sealed class ReportRow
{
    public required string FilePath { get; init; }
    public required int StatementCount { get; init; }
    public required int SucceededStatements { get; init; }
    public required int FailedStatements { get; init; }
    public required int? FirstErrorOffset { get; init; }
    public required string FirstErrorMessage { get; init; }
    public required string ElapsedMs { get; init; }
}

public sealed class ErrorReportRow
{
    public required string Category { get; init; }
    public required string ErrorMessage { get; init; }
    public required string FilePath { get; init; }
    public required string FirstErrorOffset { get; init; }
    public required int ErrorLine { get; init; }
    public required int StatementStartLine { get; init; }
    public required string StatementKind { get; init; }
    public required string StatementPreview { get; init; }
    public required string ContextPreview { get; init; }
    public required string SuggestedTestName { get; init; }
}
