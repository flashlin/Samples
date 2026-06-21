using System.Text;

namespace T1.SqlSharpE2eParser;

public sealed class TestSyntaxCoverageAnalyzer
{
    public const string TestSyntaxCoverageFileName = "test-syntax-coverage.csv";
    public const string FeatureMatrixFileName = "tsql-feature-matrix.csv";
    public const string FeaturePriorityFileName = "feature-priority.csv";

    public TestSyntaxCoverageResult Analyze(string testPath, string outputPath)
    {
        Directory.CreateDirectory(outputPath);
        var snippets = new TestSqlSnippetExtractor().Extract(testPath);
        var coverageRows = CreateCoverageRows(testPath, snippets);
        var corpusRows = ReadCorpusRows(Path.Combine(outputPath, ScanReportWriter.ErrorSummaryCsvFileName));
        var corpusAssignments = CreateCorpusAssignments(corpusRows, coverageRows);
        var matrixRows = CreateMatrixRows(coverageRows, corpusAssignments);
        var priorityRows = corpusAssignments
            .OrderByDescending(row => row.Score)
            .ThenByDescending(row => row.CorpusFailCount)
            .ThenByDescending(row => row.Feature.Priority)
            .ToList();

        var coveragePath = Path.Combine(outputPath, TestSyntaxCoverageFileName);
        var matrixPath = Path.Combine(outputPath, FeatureMatrixFileName);
        var priorityPath = Path.Combine(outputPath, FeaturePriorityFileName);

        WriteCoverageCsv(coveragePath, coverageRows);
        WriteMatrixCsv(matrixPath, matrixRows);
        WritePriorityCsv(priorityPath, priorityRows);

        return new TestSyntaxCoverageResult
        {
            TestSyntaxCoverageCsvPath = coveragePath,
            FeatureMatrixCsvPath = matrixPath,
            FeaturePriorityCsvPath = priorityPath,
            SnippetCount = snippets.Count,
            FeatureCount = TSqlFeatureCatalog.All.Count,
            CoveredFeatureCount = coverageRows.Count(row => row.TestCaseCount > 0),
            CorpusMatchedFeatureCount = matrixRows.Count(row => row.CorpusFailCount > 0)
        };
    }

    private static List<FeatureCoverageRow> CreateCoverageRows(string testPath, IReadOnlyList<TestSqlSnippet> snippets)
    {
        return TSqlFeatureCatalog.All
            .Select(feature =>
            {
                var matched = snippets
                    .Where(snippet => feature.IsMatch(snippet.Sql))
                    .ToList();
                var representative = matched.FirstOrDefault();
                return new FeatureCoverageRow
                {
                    Feature = feature,
                    TestCaseCount = matched.Count,
                    RepresentativeTestFile = representative == null ? string.Empty : Path.GetRelativePath(testPath, representative.FilePath),
                    RepresentativeLine = representative?.LineNumber ?? 0,
                    RepresentativeSnippet = representative == null ? string.Empty : NormalizePreview(representative.Sql)
                };
            })
            .OrderBy(row => row.Feature.Category, StringComparer.OrdinalIgnoreCase)
            .ThenBy(row => row.Feature.FeatureId, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<CorpusErrorSummaryRow> ReadCorpusRows(string errorSummaryPath)
    {
        if (!File.Exists(errorSummaryPath))
        {
            return [];
        }

        return File.ReadLines(errorSummaryPath)
            .Skip(1)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(ParseCsvLine)
            .Where(fields => fields.Count >= 8)
            .Select(fields => new CorpusErrorSummaryRow
            {
                Category = fields[0],
                ErrorMessage = fields[1],
                StatementKind = fields[2],
                Count = ParseInt(fields[3]),
                RepresentativeFile = fields[4],
                RepresentativeLine = fields[5],
                RepresentativePreview = fields[6],
                SuggestedTestName = fields[7]
            })
            .ToList();
    }

    private static List<CorpusPriorityRow> CreateCorpusAssignments(
        IReadOnlyList<CorpusErrorSummaryRow> corpusRows,
        IReadOnlyList<FeatureCoverageRow> coverageRows)
    {
        var testCounts = coverageRows.ToDictionary(row => row.Feature.FeatureId, row => row.TestCaseCount);
        return corpusRows
            .Select(row =>
            {
                var feature = ChooseFeature(row);
                var testCaseCount = testCounts.GetValueOrDefault(feature.FeatureId);
                return new CorpusPriorityRow
                {
                    Feature = feature,
                    Category = row.Category,
                    ErrorMessage = row.ErrorMessage,
                    StatementKind = row.StatementKind,
                    CorpusFailCount = row.Count,
                    RepresentativeCorpusFile = row.RepresentativeFile,
                    RepresentativeCorpusLine = row.RepresentativeLine,
                    RepresentativeCorpusPreview = NormalizePreview(row.RepresentativePreview),
                    CoveredByTests = testCaseCount > 0,
                    TestCaseCount = testCaseCount,
                    NextAction = DecideNextAction(testCaseCount, row.Count),
                    Score = CalculateScore(feature.Priority, testCaseCount, row.Count)
                };
            })
            .ToList();
    }

    private static List<FeatureMatrixRow> CreateMatrixRows(
        IReadOnlyList<FeatureCoverageRow> coverageRows,
        IReadOnlyList<CorpusPriorityRow> corpusAssignments)
    {
        return coverageRows
            .Select(coverage =>
            {
                var matchedCorpusRows = corpusAssignments
                    .Where(row => row.Feature.FeatureId == coverage.Feature.FeatureId)
                    .ToList();
                var representative = matchedCorpusRows.OrderByDescending(row => row.CorpusFailCount).FirstOrDefault();
                var failCount = matchedCorpusRows.Sum(row => row.CorpusFailCount);
                return new FeatureMatrixRow
                {
                    Feature = coverage.Feature,
                    TestCaseCount = coverage.TestCaseCount,
                    RepresentativeTestFile = coverage.RepresentativeTestFile,
                    RepresentativeTestLine = coverage.RepresentativeLine,
                    CorpusFailCount = failCount,
                    RepresentativeCorpusFile = representative?.RepresentativeCorpusFile ?? string.Empty,
                    RepresentativeCorpusLine = representative?.RepresentativeCorpusLine ?? string.Empty,
                    RepresentativeCorpusPreview = representative?.RepresentativeCorpusPreview ?? string.Empty,
                    NextAction = DecideNextAction(coverage.TestCaseCount, failCount),
                    Score = CalculateScore(coverage.Feature.Priority, coverage.TestCaseCount, failCount)
                };
            })
            .OrderByDescending(row => row.CorpusFailCount)
            .ThenBy(row => row.Feature.Category, StringComparer.OrdinalIgnoreCase)
            .ThenBy(row => row.Feature.FeatureId, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static TSqlFeatureDefinition ChooseFeature(CorpusErrorSummaryRow row)
    {
        var previewFeature = ChooseByRepresentativePreview(row);
        if (previewFeature != null)
        {
            return previewFeature;
        }

        var exactCategoryMatches = TSqlFeatureCatalog.All
            .Where(feature => feature.CorpusCategories.Any(category => row.Category.Contains(category, StringComparison.OrdinalIgnoreCase)))
            .Where(feature => !IsBroadCategory(row.Category))
            .ToList();
        var exactCategoryMatch = ChooseBestFeature(row, exactCategoryMatches);
        if (exactCategoryMatch != null)
        {
            return exactCategoryMatch;
        }

        var statementFeature = ChooseByStatementKind(row);
        if (statementFeature != null)
        {
            return statementFeature;
        }

        var textMatches = TSqlFeatureCatalog.All
            .Where(feature => feature.IsMatch(row.RepresentativePreview))
            .ToList();
        return ChooseBestFeature(row, textMatches) ?? TSqlFeatureCatalog.All.First(feature => feature.FeatureId == "select_statement");
    }

    private static TSqlFeatureDefinition? ChooseByRepresentativePreview(CorpusErrorSummaryRow row)
    {
        var preview = row.RepresentativePreview;
        if (preview.Contains("CREATE TYPE", StringComparison.OrdinalIgnoreCase)
            && preview.Contains(" AS TABLE", StringComparison.OrdinalIgnoreCase))
        {
            return FindFeature("create_type_table");
        }

        if (preview.Contains("DECLARE @", StringComparison.OrdinalIgnoreCase)
            && preview.Contains(" TABLE ", StringComparison.OrdinalIgnoreCase))
        {
            return FindFeature("declare_table_variable");
        }

        return null;
    }

    private static bool IsBroadCategory(string category)
    {
        return category.Equals("Parser returned null result", StringComparison.OrdinalIgnoreCase)
               || category.Equals("Permission statement variant", StringComparison.OrdinalIgnoreCase)
               || category.Equals("CREATE PROCEDURE variant", StringComparison.OrdinalIgnoreCase)
               || category.Equals("Unclassified parser failure", StringComparison.OrdinalIgnoreCase);
    }

    private static TSqlFeatureDefinition? ChooseByStatementKind(CorpusErrorSummaryRow row)
    {
        var statementKind = row.StatementKind.ToUpperInvariant();
        if (statementKind.StartsWith("CREATE TABLE"))
        {
            return FindFeature("create_table");
        }

        if (statementKind.StartsWith("CREATE PROCEDURE"))
        {
            return FindFeature("create_procedure");
        }

        if (statementKind.StartsWith("CREATE VIEW"))
        {
            return FindFeature("create_view");
        }

        if (statementKind.StartsWith("CREATE CLUSTERED"))
        {
            return FindFeature("create_index");
        }

        if (statementKind.StartsWith("DECLARE"))
        {
            return row.RepresentativePreview.Contains(" TABLE ", StringComparison.OrdinalIgnoreCase)
                ? FindFeature("declare_table_variable")
                : FindFeature("declare_scalar");
        }

        if (statementKind.StartsWith("IF"))
        {
            return FindFeature("if_statement");
        }

        if (statementKind.StartsWith("ELSE"))
        {
            return FindFeature("else_statement");
        }

        if (statementKind.StartsWith("INSERT"))
        {
            if (row.RepresentativePreview.Contains("VALUES", StringComparison.OrdinalIgnoreCase))
            {
                return FindFeature("insert_multi_values");
            }

            if (row.RepresentativePreview.Contains("SELECT", StringComparison.OrdinalIgnoreCase))
            {
                return FindFeature("insert_select");
            }

            return FindFeature("insert_without_into");
        }

        if (statementKind.StartsWith("UPDATE"))
        {
            return row.RepresentativePreview.Contains(" WITH", StringComparison.OrdinalIgnoreCase)
                ? FindFeature("update_with_hints")
                : FindFeature("update_statement");
        }

        if (statementKind.StartsWith("DELETE"))
        {
            return FindFeature("delete_statement");
        }

        if (statementKind.StartsWith("SELECT"))
        {
            return row.RepresentativePreview.Contains("@", StringComparison.OrdinalIgnoreCase)
                ? FindFeature("select_assignment")
                : FindFeature("select_statement");
        }

        if (statementKind.StartsWith("EXEC"))
        {
            return FindFeature("exec_statement");
        }

        if (statementKind.StartsWith("USE") || statementKind.StartsWith("SET ANSI"))
        {
            return FindFeature("use_go_batch");
        }

        return null;
    }

    private static TSqlFeatureDefinition? ChooseBestFeature(CorpusErrorSummaryRow row, IReadOnlyList<TSqlFeatureDefinition> features)
    {
        if (features.Count == 0)
        {
            return null;
        }

        var preview = row.RepresentativePreview;
        return features
            .OrderByDescending(feature => feature.IsMatch(preview))
            .ThenByDescending(feature => feature.Priority)
            .First();
    }

    private static TSqlFeatureDefinition? FindFeature(string featureId)
    {
        return TSqlFeatureCatalog.All.FirstOrDefault(feature => feature.FeatureId == featureId);
    }

    private static string DecideNextAction(int testCaseCount, int corpusFailCount)
    {
        if (corpusFailCount == 0 && testCaseCount > 0)
        {
            return "Covered";
        }

        if (corpusFailCount == 0)
        {
            return "Add checklist test";
        }

        if (testCaseCount == 0)
        {
            return "Add unit test then parser support";
        }

        return "Extend parser support";
    }

    private static int CalculateScore(int priority, int testCaseCount, int corpusFailCount)
    {
        var missingTestBonus = testCaseCount == 0 ? 50 : 0;
        return corpusFailCount * 10 + priority + missingTestBonus;
    }

    private static void WriteCoverageCsv(string path, IReadOnlyList<FeatureCoverageRow> rows)
    {
        var csv = new StringBuilder();
        csv.AppendLine("FeatureId,Category,Feature,Priority,TestCaseCount,RepresentativeTestFile,RepresentativeLine,RepresentativeSnippet");
        foreach (var row in rows)
        {
            csv.AppendLine(string.Join(",", [
                ScanReportWriter.EscapeCsv(row.Feature.FeatureId),
                ScanReportWriter.EscapeCsv(row.Feature.Category),
                ScanReportWriter.EscapeCsv(row.Feature.Feature),
                row.Feature.Priority.ToString(),
                row.TestCaseCount.ToString(),
                ScanReportWriter.EscapeCsv(row.RepresentativeTestFile),
                row.RepresentativeLine.ToString(),
                ScanReportWriter.EscapeCsv(row.RepresentativeSnippet)
            ]));
        }
        File.WriteAllText(path, csv.ToString(), Encoding.UTF8);
    }

    private static void WriteMatrixCsv(string path, IReadOnlyList<FeatureMatrixRow> rows)
    {
        var csv = new StringBuilder();
        csv.AppendLine("FeatureId,Category,Feature,Priority,CoveredByTests,TestCaseCount,CorpusFailCount,Score,NextAction,RepresentativeTestFile,RepresentativeTestLine,RepresentativeCorpusFile,RepresentativeCorpusLine,RepresentativeCorpusPreview");
        foreach (var row in rows)
        {
            csv.AppendLine(string.Join(",", [
                ScanReportWriter.EscapeCsv(row.Feature.FeatureId),
                ScanReportWriter.EscapeCsv(row.Feature.Category),
                ScanReportWriter.EscapeCsv(row.Feature.Feature),
                row.Feature.Priority.ToString(),
                (row.TestCaseCount > 0).ToString(),
                row.TestCaseCount.ToString(),
                row.CorpusFailCount.ToString(),
                row.Score.ToString(),
                ScanReportWriter.EscapeCsv(row.NextAction),
                ScanReportWriter.EscapeCsv(row.RepresentativeTestFile),
                row.RepresentativeTestLine.ToString(),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusFile),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusLine),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusPreview)
            ]));
        }
        File.WriteAllText(path, csv.ToString(), Encoding.UTF8);
    }

    private static void WritePriorityCsv(string path, IReadOnlyList<CorpusPriorityRow> rows)
    {
        var csv = new StringBuilder();
        csv.AppendLine("Rank,ErrorCategory,ErrorMessage,StatementKind,FeatureId,FeatureCategory,Feature,Priority,CoveredByTests,TestCaseCount,CorpusFailCount,Score,NextAction,RepresentativeCorpusFile,RepresentativeCorpusLine,RepresentativeCorpusPreview");
        var rank = 1;
        foreach (var row in rows)
        {
            csv.AppendLine(string.Join(",", [
                rank.ToString(),
                ScanReportWriter.EscapeCsv(row.Category),
                ScanReportWriter.EscapeCsv(row.ErrorMessage),
                ScanReportWriter.EscapeCsv(row.StatementKind),
                ScanReportWriter.EscapeCsv(row.Feature.FeatureId),
                ScanReportWriter.EscapeCsv(row.Feature.Category),
                ScanReportWriter.EscapeCsv(row.Feature.Feature),
                row.Feature.Priority.ToString(),
                (row.TestCaseCount > 0).ToString(),
                row.TestCaseCount.ToString(),
                row.CorpusFailCount.ToString(),
                row.Score.ToString(),
                ScanReportWriter.EscapeCsv(row.NextAction),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusFile),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusLine),
                ScanReportWriter.EscapeCsv(row.RepresentativeCorpusPreview)
            ]));
            rank++;
        }
        File.WriteAllText(path, csv.ToString(), Encoding.UTF8);
    }

    private static string NormalizePreview(string value)
    {
        var normalized = string.Join(" ", value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        return normalized.Length <= 240 ? normalized : normalized[..240];
    }

    private static int ParseInt(string value)
    {
        return int.TryParse(value, out var result) ? result : 0;
    }

    private static List<string> ParseCsvLine(string line)
    {
        var fields = new List<string>();
        var field = new StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var ch = line[i];
            if (inQuotes)
            {
                if (ch == '"' && i + 1 < line.Length && line[i + 1] == '"')
                {
                    field.Append('"');
                    i++;
                    continue;
                }

                if (ch == '"')
                {
                    inQuotes = false;
                    continue;
                }

                field.Append(ch);
                continue;
            }

            if (ch == '"')
            {
                inQuotes = true;
                continue;
            }

            if (ch == ',')
            {
                fields.Add(field.ToString());
                field.Clear();
                continue;
            }

            field.Append(ch);
        }

        fields.Add(field.ToString());
        return fields;
    }
}

public sealed record TestSyntaxCoverageResult
{
    public required string TestSyntaxCoverageCsvPath { get; init; }
    public required string FeatureMatrixCsvPath { get; init; }
    public required string FeaturePriorityCsvPath { get; init; }
    public required int SnippetCount { get; init; }
    public required int FeatureCount { get; init; }
    public required int CoveredFeatureCount { get; init; }
    public required int CorpusMatchedFeatureCount { get; init; }
}

public sealed record FeatureCoverageRow
{
    public required TSqlFeatureDefinition Feature { get; init; }
    public required int TestCaseCount { get; init; }
    public required string RepresentativeTestFile { get; init; }
    public required int RepresentativeLine { get; init; }
    public required string RepresentativeSnippet { get; init; }
}

public sealed record CorpusErrorSummaryRow
{
    public required string Category { get; init; }
    public required string ErrorMessage { get; init; }
    public required string StatementKind { get; init; }
    public required int Count { get; init; }
    public required string RepresentativeFile { get; init; }
    public required string RepresentativeLine { get; init; }
    public required string RepresentativePreview { get; init; }
    public required string SuggestedTestName { get; init; }
}

public sealed record FeatureMatrixRow
{
    public required TSqlFeatureDefinition Feature { get; init; }
    public required int TestCaseCount { get; init; }
    public required string RepresentativeTestFile { get; init; }
    public required int RepresentativeTestLine { get; init; }
    public required int CorpusFailCount { get; init; }
    public required int Score { get; init; }
    public required string NextAction { get; init; }
    public required string RepresentativeCorpusFile { get; init; }
    public required string RepresentativeCorpusLine { get; init; }
    public required string RepresentativeCorpusPreview { get; init; }
}

public sealed record CorpusPriorityRow
{
    public required TSqlFeatureDefinition Feature { get; init; }
    public required string Category { get; init; }
    public required string ErrorMessage { get; init; }
    public required string StatementKind { get; init; }
    public required bool CoveredByTests { get; init; }
    public required int TestCaseCount { get; init; }
    public required int CorpusFailCount { get; init; }
    public required int Score { get; init; }
    public required string NextAction { get; init; }
    public required string RepresentativeCorpusFile { get; init; }
    public required string RepresentativeCorpusLine { get; init; }
    public required string RepresentativeCorpusPreview { get; init; }
}
