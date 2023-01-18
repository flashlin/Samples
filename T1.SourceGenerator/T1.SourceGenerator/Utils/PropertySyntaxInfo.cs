using Microsoft.CodeAnalysis;

namespace T1.SourceGenerator.Utils;

public class PropertySyntaxInfo
{
    public Accessibility Accessibility { get; set; }
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
    public bool HasGetter { get; set; }
    public bool HasSetter { get; set; }
}