namespace T1.SourceGenerator.ApiClientGen;

public class WebApiMethodContext
{
    public string MethodReturnTypeFullName { get; set; }
    public string MethodName { get; set; }
    public string MethodArguments { get; set; }
    public string MethodParameters { get; set; }
    public string? ApiPath { get; set; }
}