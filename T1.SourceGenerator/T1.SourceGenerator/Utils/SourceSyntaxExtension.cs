using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator.Utils;

public static class SourceSyntaxExtension
{
    public static List<TypeSyntaxInfo> GetAllTypes(this Compilation compilation)
    {
        return compilation
            .SyntaxTrees
            .SelectMany(x => x.GetRoot().DescendantNodes().OfType<TypeDeclarationSyntax>())
            .Select(x => new TypeSyntaxInfo
            {
                TypeFullName = x.GetFullName(),
                SyntaxNode = x,
            })
            .ToList();
    }

    public static List<AttributeData> GetAttributeDataList(this AttributeListSyntax attributes, Compilation compilation)
    {
        var acceptedTrees = new HashSet<SyntaxTree>();
        foreach (var attribute in attributes.Attributes)
            acceptedTrees.Add(attribute.SyntaxTree);

        var parentSymbol = attributes.Parent!.GetDeclaredSymbol(compilation)!;
        var parentAttributes = parentSymbol.GetAttributes();
        var ret = new List<AttributeData>();
        foreach (var attribute in parentAttributes)
        {
            if (acceptedTrees.Contains(attribute.ApplicationSyntaxReference!.SyntaxTree))
                ret.Add(attribute);
        }

        return ret;
    }

    public static IEnumerable<AttributeData> GetAttributeDataList(this TypeDeclarationSyntax node,
        Compilation compilation)
    {
        return node.AttributeLists
            .SelectMany(x => x.GetAttributeDataList(compilation));
    }

    public static IEnumerable<PropertySyntaxInfo> GetPropertiesSyntaxList(this SyntaxNode typeSyntaxNode,
        Compilation compilation)
    {
        var model = compilation.GetSemanticModel(typeSyntaxNode.SyntaxTree);
        var symbol = (model.GetDeclaredSymbol(typeSyntaxNode) as INamedTypeSymbol)!;
        foreach (var member in symbol.GetMembers())
        {
            if (member.Kind == SymbolKind.Property)
            {
                var property = (IPropertySymbol)member;
                yield return new PropertySyntaxInfo
                {
                    Accessibility = property.DeclaredAccessibility,
                    TypeFullName = property.Type.ToDisplayString(),
                    Name = property.Name,
                    HasGetter = property.GetMethod != null,
                    HasSetter = property.SetMethod != null,
                };
            }
        }
    }
}

public class PropertySyntaxInfo
{
    public Accessibility Accessibility { get; set; }
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
    public bool HasGetter { get; set; }
    public bool HasSetter { get; set; }
}

public class FileContentInfo
{
    public string Directory { get; set; }
    public string Content { get; set; }
}