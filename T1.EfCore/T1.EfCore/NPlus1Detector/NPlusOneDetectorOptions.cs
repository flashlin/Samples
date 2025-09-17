namespace T1.EfCore.NPlus1Detector;

public class NPlusOneDetectorOptions
{
    public int Threshold { get; init; } = 3;
    public int DetectionWindowMs { get; init; } = 2000;
    public int CooldownMs { get; init; } = 3000;
    public bool CaptureStackTrace { get; init; } = true;
    public bool LogToConsole { get; init; } = true;
    public int CleanupIntervalMinutes { get; init; } = 5;
    public Action<NPlusOneDetectionResult>? OnDetection { get; init; }
}

public class NPlusOneDetectionResult
{
    public string Query { get; init; } = string.Empty;
    public int ExecutionCount { get; init; }
    public double DurationMs { get; init; }
    public string? StackTrace { get; init; }
    public string? DbContextType { get; init; }
    public DateTime DetectedAt { get; init; } = DateTime.UtcNow;
}