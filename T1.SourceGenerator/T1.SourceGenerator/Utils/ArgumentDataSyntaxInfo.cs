namespace T1.SourceGenerator.Utils;

public class ArgumentDataSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public string ValueTypeFullName { get; set; } = null!;
    public object? Value { get; set; }
}