using Microsoft.CodeAnalysis;
using T1.SourceGenerator.Attributes;

namespace T1.SourceGenerator.Utils;

public class RoslynGeneratorExecutionContext : IRoslynGeneratorExecutionContext
{
    private readonly GeneratorExecutionContext _context;

    public RoslynGeneratorExecutionContext(GeneratorExecutionContext context)
    {
        _context = context;
    }

    public List<TypeSyntaxInfo> AllTypes { get; set; } = new();
    public List<EnumSyntaxInfo> AllEnums { get; set; } = new();

    public void AddSource(string name, string sourceText)
    {
        _context.AddSource(name, sourceText);
    }
}
