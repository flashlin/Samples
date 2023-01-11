using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator.Utils;

public class TypeSyntaxInfo
{
    public string TypeFullName { get; set; } = null!;
    public TypeDeclarationSyntax SyntaxNode { get; set; } = null!;
}