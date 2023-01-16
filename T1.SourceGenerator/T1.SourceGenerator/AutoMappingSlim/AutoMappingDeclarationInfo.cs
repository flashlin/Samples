using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator.AutoMappingSlim;

public class AutoMappingDeclarationInfo
{
    public string ToTypeFullName { get; set; } = null!;
    public TypeDeclarationSyntax ToTypeSyntax { get; set; } = null!;
}