namespace T1.SourceGenerator.Utils;

public class AttributeSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public List<ArgumentSyntaxInfo> ConstructorArguments { get; set; } = null!;

    public ArgumentSyntaxInfo? GetArgumentSyntaxInfo(string name)
    {
        return ConstructorArguments
            .FirstOrDefault(x => x.Name == name);
    }

    public override string ToString()
    {
        return $"{TypeFullName}";
    }
}