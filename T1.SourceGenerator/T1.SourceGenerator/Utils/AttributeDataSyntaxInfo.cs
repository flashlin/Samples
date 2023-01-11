namespace T1.SourceGenerator.Utils;

public class AttributeDataSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public List<ArgumentDataSyntaxInfo> ConstructorArguments { get; set; } = null!;
}