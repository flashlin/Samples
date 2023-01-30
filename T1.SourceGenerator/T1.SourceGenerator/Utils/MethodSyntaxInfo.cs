namespace T1.SourceGenerator.Utils;

public class MethodSyntaxInfo
{
    public string Name { get; set; } = null!;
    public List<ParameterSyntaxInfo> Parameters { get; set; } = new();
    public List<AttributeSyntaxInfo> Attributes { get; set; } = new();
    public string ReturnTypeFullName { get; set; } = null!;
    public string BodySourceCode { get; set; } = string.Empty;
}