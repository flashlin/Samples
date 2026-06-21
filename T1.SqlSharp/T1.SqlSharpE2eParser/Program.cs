using System.Text.Json;

namespace T1.SqlSharpE2eParser;

public static class Program
{
    private const string DefaultSourcePath = "/Users/flash/titan/DbProjects";
    private const string ScanFileCommand = "--scan-file";
    private const string WorkerCommand = "--worker";
    private const string AnalyzeReportCommand = "--analyze-report";
    private const string AnalyzeTestCoverageCommand = "--analyze-test-coverage";

    public static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == ScanFileCommand)
        {
            return ScanFile(args);
        }

        if (args.Length > 0 && args[0] == WorkerCommand)
        {
            return RunWorker(args);
        }

        if (args.Length > 0 && args[0] == AnalyzeReportCommand)
        {
            return AnalyzeReport(args);
        }

        if (args.Length > 0 && args[0] == AnalyzeTestCoverageCommand)
        {
            return AnalyzeTestCoverage(args);
        }

        var sourcePath = args.Length > 0 ? args[0] : DefaultSourcePath;
        var outputPath = args.Length > 1 ? args[1] : ResolveDefaultOutputPath();

        if (!Directory.Exists(sourcePath))
        {
            Console.Error.WriteLine($"Source path not found: {sourcePath}");
            return 1;
        }

        var scanner = new SqlCorpusScanner();
        var report = scanner.Scan(sourcePath, outputPath);
        ScanReportWriter.Write(report, outputPath);

        Console.WriteLine($"JSON report: {Path.Combine(outputPath, ScanReportWriter.JsonFileName)}");
        Console.WriteLine($"CSV report: {Path.Combine(outputPath, ScanReportWriter.CsvFileName)}");
        return 0;
    }

    private static int AnalyzeReport(string[] args)
    {
        var sourcePath = args.Length > 1 ? args[1] : DefaultSourcePath;
        var outputPath = args.Length > 2 ? args[2] : ResolveDefaultOutputPath();

        if (!Directory.Exists(sourcePath))
        {
            Console.Error.WriteLine($"Source path not found: {sourcePath}");
            return 1;
        }

        var analyzer = new ReportErrorAnalyzer();
        var result = analyzer.Analyze(sourcePath, outputPath);
        Console.WriteLine($"Error report: {result.ErrorCsvPath}");
        Console.WriteLine($"Error summary: {result.ErrorSummaryCsvPath}");
        return 0;
    }

    private static int AnalyzeTestCoverage(string[] args)
    {
        var testPath = args.Length > 1 ? args[1] : ResolveDefaultTestPath();
        var outputPath = args.Length > 2 ? args[2] : ResolveDefaultOutputPath();

        if (!Directory.Exists(testPath))
        {
            Console.Error.WriteLine($"Test path not found: {testPath}");
            return 1;
        }

        var analyzer = new TestSyntaxCoverageAnalyzer();
        var result = analyzer.Analyze(testPath, outputPath);
        Console.WriteLine($"SQL snippets: {result.SnippetCount}");
        Console.WriteLine($"Features: {result.CoveredFeatureCount}/{result.FeatureCount}");
        Console.WriteLine($"Corpus matched features: {result.CorpusMatchedFeatureCount}");
        Console.WriteLine($"Test syntax coverage: {result.TestSyntaxCoverageCsvPath}");
        Console.WriteLine($"Feature matrix: {result.FeatureMatrixCsvPath}");
        Console.WriteLine($"Feature priority: {result.FeaturePriorityCsvPath}");
        return 0;
    }

    private static int ScanFile(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("Usage: --scan-file <sourcePath> <filePath>");
            return 1;
        }

        var result = SqlCorpusScanner.ScanFileInProcess(args[1], args[2]);
        Console.WriteLine(JsonSerializer.Serialize(result));
        return 0;
    }

    private static int RunWorker(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("Usage: --worker <sourcePath>");
            return 1;
        }

        var sourcePath = args[1];
        string? filePath;
        while ((filePath = Console.ReadLine()) != null)
        {
            var result = SqlCorpusScanner.ScanFileInProcess(sourcePath, filePath);
            Console.WriteLine(JsonSerializer.Serialize(result));
            Console.Out.Flush();
        }

        return 0;
    }

    private static string ResolveDefaultOutputPath()
    {
        var currentPath = Directory.GetCurrentDirectory();
        if (Path.GetFileName(currentPath) == "T1.SqlSharpE2eParser")
        {
            return Path.Combine(currentPath, "out");
        }

        var projectPath = Path.Combine(currentPath, "T1.SqlSharpE2eParser");
        if (Directory.Exists(projectPath))
        {
            return Path.Combine(projectPath, "out");
        }

        return Path.Combine(currentPath, "out");
    }

    private static string ResolveDefaultTestPath()
    {
        var currentPath = Directory.GetCurrentDirectory();
        if (Path.GetFileName(currentPath) == "T1.SqlSharpTests")
        {
            return currentPath;
        }

        var testPath = Path.Combine(currentPath, "T1.SqlSharpTests");
        if (Directory.Exists(testPath))
        {
            return testPath;
        }

        return currentPath;
    }
}
