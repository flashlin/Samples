using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator.Utils;

public class TypeSyntaxInfo
{
    public List<AttributeSyntaxInfo> Attributes { get; set; } = new();
    public string TypeFullName { get; set; } = null!;
    public List<MethodSyntaxInfo> Methods { get; set; } = new();
    public TypeDeclarationSyntax SyntaxNode { get; set; } = null!;

    public override string ToString()
    {
        return $"{TypeFullName}";
    }
}