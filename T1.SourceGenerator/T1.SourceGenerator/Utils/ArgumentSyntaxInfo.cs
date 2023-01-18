namespace T1.SourceGenerator.Utils;

public class ArgumentSyntaxInfo
{
    public string Name { get; set; } = null!;
    public string TypeFullName { get; set; } = null!;
    public string ValueTypeFullName { get; set; } = null!;
    public object? Value { get; set; }

    public override string ToString()
    {
        return $"{TypeFullName} {Name}";
    }
}