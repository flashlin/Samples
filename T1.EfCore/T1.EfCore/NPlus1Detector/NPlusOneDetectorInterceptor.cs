namespace T1.EfCore.NPlus1Detector;

public sealed class NPlusOneDetectorInterceptor : DbCommandInterceptor, IDisposable
{
    private readonly ILogger<NPlusOneDetectorInterceptor> _logger;
    private readonly NPlusOneDetectorOptions _options;
    private readonly ConcurrentDictionary<string, QueryPattern> _queryPatterns = new();
    private readonly Timer _cleanupTimer;
    private bool _disposed;

    public NPlusOneDetectorInterceptor(
        ILogger<NPlusOneDetectorInterceptor> logger,
        NPlusOneDetectorOptions options)
    {
        _logger = logger;
        _options = options;

        var cleanupInterval = TimeSpan.FromMinutes(_options.CleanupIntervalMinutes);
        _cleanupTimer = new Timer(CleanupOldPatterns, null, cleanupInterval, cleanupInterval);
    }

    public override InterceptionResult<DbDataReader> ReaderExecuting(
        DbCommand command,
        CommandEventData eventData,
        InterceptionResult<DbDataReader> result)
    {
        AnalyzeQuery(command, eventData);
        return base.ReaderExecuting(command, eventData, result);
    }

    public override ValueTask<InterceptionResult<DbDataReader>> ReaderExecutingAsync(
        DbCommand command,
        CommandEventData eventData,
        InterceptionResult<DbDataReader> result,
        CancellationToken cancellationToken = default)
    {
        AnalyzeQuery(command, eventData);
        return base.ReaderExecutingAsync(command, eventData, result, cancellationToken);
    }

    private void AnalyzeQuery(DbCommand command, CommandEventData eventData)
    {
        var sql = command.CommandText.Trim();

        if (string.IsNullOrEmpty(sql) || !sql.StartsWith("SELECT", StringComparison.OrdinalIgnoreCase))
            return;

        var normalizedQuery = NormalizeQuery(sql);
        var now = DateTime.UtcNow;
        var stackTrace = _options.CaptureStackTrace ? GetCleanStackTrace() : null;

        var pattern = _queryPatterns.AddOrUpdate(
            normalizedQuery,
            new QueryPattern
            {
                FirstSeen = now,
                LastSeen = now,
                Count = 1,
                OriginalSql = sql,
                StackTrace = stackTrace,
                DbContextName = eventData.Context?.GetType().Name,
                LastAlerted = DateTime.MinValue
            },
            (_, existing) =>
            {
                existing.Count++;
                existing.LastSeen = now;
                return existing;
            });

        var detectionWindow = TimeSpan.FromMilliseconds(_options.DetectionWindowMs);
        var cooldownPeriod = TimeSpan.FromMilliseconds(_options.CooldownMs);

        if (pattern.Count >= _options.Threshold &&
            (pattern.LastSeen - pattern.FirstSeen) <= detectionWindow &&
            (now - pattern.LastAlerted) >= cooldownPeriod)
        {
            HandleDetection(pattern);

            pattern.Count = 1;
            pattern.FirstSeen = now;
            pattern.LastAlerted = now;
        }
    }

    private void HandleDetection(QueryPattern pattern)
    {
        var duration = (pattern.LastSeen - pattern.FirstSeen).TotalMilliseconds;
        var query = pattern.OriginalSql.Length > 150
            ? pattern.OriginalSql.Substring(0, 150) + "..."
            : pattern.OriginalSql;

        var result = new NPlusOneDetectionResult
        {
            Query = query,
            ExecutionCount = pattern.Count,
            DurationMs = duration,
            StackTrace = pattern.StackTrace,
            DbContextType = pattern.DbContextName,
            DetectedAt = DateTime.UtcNow
        };

        if (_options.LogToConsole)
        {
            _logger.LogWarning(
                "N+1 Query Detected: {Count} executions in {Duration}ms\n" +
                "Context: {DbContext}\n" +
                "Query: {Query}\n" +
                "Stack Trace:\n{StackTrace}",
                result.ExecutionCount,
                result.DurationMs,
                result.DbContextType,
                result.Query,
                result.StackTrace ?? "Not captured");
        }

        try
        {
            _options.OnDetection?.Invoke(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in N+1 detection callback");
        }
    }

    private string NormalizeQuery(string sql)
    {
        if (string.IsNullOrWhiteSpace(sql))
            return string.Empty;

        var normalized = sql.Trim();

        // Replace string literals first to avoid affecting them
        // Handles: 'text', 'text with ''escaped'' quotes'
        normalized = Regex.Replace(
            normalized,
            @"'(?:[^']|'')*'",
            "'?'",
            RegexOptions.Compiled);

        // Replace all parameter variations
        // Handles: @param, :param, $1, ?
        normalized = Regex.Replace(
            normalized,
            @"@\w+|:\w+|\$\d+|\?",
            "?",
            RegexOptions.Compiled);

        // Replace numeric literals
        // Handles: 123, 45.67, -89
        normalized = Regex.Replace(
            normalized,
            @"-?\b\d+(?:\.\d+)?\b",
            "?",
            RegexOptions.Compiled);

        // Normalize IN clauses with multiple values
        // Handles: IN (1,2,3), IN (@p1, @p2, @p3), IN (1, 2, 'text')
        normalized = Regex.Replace(
            normalized,
            @"\bIN\s*\(\s*[^)]+\s*\)",
            "IN (?)",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Normalize VALUES clauses for INSERT statements
        // Handles: VALUES (1,2), (3,4), (5,6)
        normalized = Regex.Replace(
            normalized,
            @"\bVALUES\s*\(\s*[^)]+\s*\)(?:\s*,\s*\(\s*[^)]+\s*\))*",
            "VALUES (?)",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Normalize LIMIT/OFFSET clauses
        // Handles: LIMIT 10, OFFSET 20, LIMIT 5 OFFSET 15
        normalized = Regex.Replace(
            normalized,
            @"\bLIMIT\s+\d+",
            "LIMIT ?",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        normalized = Regex.Replace(
            normalized,
            @"\bOFFSET\s+\d+",
            "OFFSET ?",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Normalize multiple whitespace characters into single spaces
        normalized = Regex.Replace(
            normalized,
            @"\s+",
            " ",
            RegexOptions.Compiled);

        // Remove extra whitespace around common SQL operators and keywords
        normalized = Regex.Replace(
            normalized,
            @"\s*([(),=<>!])\s*",
            "$1",
            RegexOptions.Compiled);

        return normalized.Trim().ToUpperInvariant();
    }

    private string GetCleanStackTrace()
    {
        var stackTrace = new StackTrace(true);
        var frames = stackTrace.GetFrames();
        var relevantFrames = new List<string>();

        if (frames.Length == 0) return "No stack trace available";

        foreach (var frame in frames)
        {
            var method = frame.GetMethod();
            if (method == null) continue;

            var typeName = method.DeclaringType?.FullName ?? "";

            if (typeName.StartsWith("Microsoft.EntityFrameworkCore") ||
                typeName.StartsWith("System.") ||
                typeName.Contains(nameof(NPlusOneDetectorInterceptor)))
                continue;

            var fileName = frame.GetFileName();
            var lineNumber = frame.GetFileLineNumber();

            var frameInfo = $"{typeName}.{method.Name}()";
            if (!string.IsNullOrEmpty(fileName) && lineNumber > 0)
            {
                frameInfo += $" in {Path.GetFileName(fileName)}:line {lineNumber}";
            }

            relevantFrames.Add(frameInfo);

            if (relevantFrames.Count >= 25) break;
        }

        return relevantFrames.Count > 0
            ? string.Join(Environment.NewLine + "   at ", relevantFrames)
            : "No relevant stack trace found";
    }

    private void CleanupOldPatterns(object? state)
    {
        var cutoff = DateTime.UtcNow.AddMinutes(-(_options.CleanupIntervalMinutes));

        var keysToRemove = _queryPatterns
            .Where(kvp => kvp.Value.LastSeen < cutoff)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var key in keysToRemove)
        {
            _queryPatterns.TryRemove(key, out _);
        }

        if (keysToRemove.Count > 0)
        {
            _logger.LogDebug("Cleaned up {Count} old query patterns", keysToRemove.Count);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _cleanupTimer.Dispose();
            _disposed = true;
        }
    }

    private class QueryPattern
    {
        public DateTime FirstSeen { get; set; }
        public DateTime LastSeen { get; set; }
        public int Count { get; set; }
        public string OriginalSql { get; init; } = string.Empty;
        public string? StackTrace { get; init; }
        public string? DbContextName { get; init; }
        public DateTime LastAlerted { get; set; } = DateTime.MinValue;
    }
}