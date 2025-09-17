using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace T1.EfCore.NPlus1Detector;

public static class DbContextOptionsBuilderExtensions
{
    /// <summary>
    /// <![CDATA[
    /// protected override void OnConfiguring(DbContextOptionsBuilder
    ///     optionsBuilder)
    /// {
    ///     optionsBuilder.AddNPlusOneDetector(new NPlusOneDetectorOptions()
    ///     {
    ///         CaptureStackTrace = true,
    ///         LogToConsole = true,
    ///         Threshold = 5,
    ///         DetectionWindowMs = 2000,
    ///         CooldownMs = 3000,
    ///         CleanupIntervalMinutes = 5,
    ///         OnDetection = (result) =>
    ///         {
    ///             Console.WriteLine($"N+1 DETECTED: {result.ExecutionCount} queries in {result.DurationMs:F2}ms");
    ///             Console.WriteLine($"Location: {result.StackTrace?.Split('\n').FirstOrDefault()?.Trim()}");
    ///             Console.WriteLine($"Query: {(result.Query.Length > 80 ? result.Query.Substring(0, 80) + "..." : result.Query)}");
    ///             Console.WriteLine($"Time: {result.DetectedAt:HH:mm:ss}");
    ///             Console.WriteLine($"Context: {result.DbContextType}");
    ///             Console.WriteLine(new string('-', 50));
    ///         }
    ///     });
    /// }
    /// ]]>
    /// </summary>
    /// <param name="optionsBuilder"></param>
    /// <param name="options"></param>
    /// <param name="logger"></param>
    /// <returns></returns>
    public static DbContextOptionsBuilder AddNPlusOneDetector(
        this DbContextOptionsBuilder optionsBuilder,
        NPlusOneDetectorOptions? options = null,
        ILogger<NPlusOneDetectorInterceptor>? logger = null)
    {
        options ??= new NPlusOneDetectorOptions();

        logger ??= CreateDefaultLogger();

        var interceptor = new NPlusOneDetectorInterceptor(logger, options);
        return optionsBuilder.AddInterceptors(interceptor);
    }

    private static ILogger<NPlusOneDetectorInterceptor> CreateDefaultLogger()
    {
        var loggerFactory = LoggerFactory.Create(builder =>
            builder.AddConsole().SetMinimumLevel(LogLevel.Warning));
        return loggerFactory.CreateLogger<NPlusOneDetectorInterceptor>();
    }
}