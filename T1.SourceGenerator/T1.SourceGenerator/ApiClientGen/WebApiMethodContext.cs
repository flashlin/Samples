namespace T1.SourceGenerator.ApiClientGen;

public class WebApiMethodContext
{
    public string MethodReturnTypeFullName { get; set; } = null!;
    public string MethodName { get; set; } = null!;
    public string MethodArguments { get; set; } = string.Empty;
    public string MethodParameters { get; set; } = string.Empty;
    public string? ApiPath { get; set; }
}