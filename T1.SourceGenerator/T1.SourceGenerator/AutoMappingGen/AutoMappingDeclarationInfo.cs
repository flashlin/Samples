using Microsoft.CodeAnalysis.CSharp.Syntax;
using T1.SourceGenerator.Attributes;

namespace T1.SourceGenerator.AutoMappingGen;

public class AutoMappingDeclarationInfo
{
    public string ToTypeFullName { get; set; } = null!;
    public TypeSyntaxInfo ToTypeSyntax { get; set; } = null!;
    public string? ToMethodName { get; set; }
}