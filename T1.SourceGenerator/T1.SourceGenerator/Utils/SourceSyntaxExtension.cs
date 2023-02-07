using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using T1.SourceGenerator.Attributes;

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
                Attributes = x.AttributeLists.QueryAttributesSyntaxInfo(compilation).ToList(),
                TypeFullName = x.GetFullName(),
                BaseTypes = x.QueryBaseTypeInfo(compilation).ToList(),
                Methods = x.QueryMethodsSyntaxInfo(compilation).ToList(),
                Properties = x.QueryPropertiesSyntaxInfo(compilation).ToList(),
            })
            .ToList();
    }


    public static string GetName(this string fullname)
    {
        var idx = fullname.LastIndexOf(".", StringComparison.Ordinal);
        if(idx == -1)
        {
            return fullname;
        }
        return fullname.Substring(idx + 1);
    }

    public static IEnumerable<string> QueryBaseTypeInfo(this TypeDeclarationSyntax typeDeclaration, Compilation compilation)
    {
        if (typeDeclaration.BaseList == null)
        {
            yield break;
        }

        foreach (var baseType in typeDeclaration.BaseList.Types)
        {
            if (baseType.Type.Kind() == SyntaxKind.IdentifierName)
            {
                yield return baseType.Type.GetTypeFullName(compilation);
            }
        }
    }
    
    public static IEnumerable<MethodSyntaxInfo> QueryMethodsSyntaxInfo(this TypeDeclarationSyntax typeDeclaration, Compilation compilation)
    {
        var methods = typeDeclaration.Members.OfType<MethodDeclarationSyntax>();
        foreach (var method in methods)
        {
            yield return new MethodSyntaxInfo
            {
                Attributes = method.QueryAttributesSyntaxInfo(compilation).ToList(),
                Name = method.Identifier.ValueText,
                Parameters = method.QueryMethodParameters(compilation).ToList(),
                BodySourceCode = (method.Body == null) ? string.Empty : method.Body.ToFullString(),
                ReturnTypeFullName = method.ReturnType.GetTypeFullName(compilation)
            };
        }
    }

    public static string GetTypeFullName(this TypeSyntax typeSyntax, Compilation compilation)
    {
        var semanticModel = compilation.GetSemanticModel(typeSyntax.SyntaxTree);
        var typeInfo = semanticModel.GetTypeInfo(typeSyntax);
        var type = (typeInfo.Type as ITypeSymbol)!;
        return type.ToDisplayString();
    }

    public static IEnumerable<AttributeSyntaxInfo> QueryAttributesSyntaxInfo(this MethodDeclarationSyntax methodDeclarationSyntax, Compilation compilation)
    {
        return methodDeclarationSyntax.AttributeLists.QueryAttributesSyntaxInfo(compilation);
    }
    
    public static IEnumerable<ParameterSyntaxInfo> QueryMethodParameters(this MethodDeclarationSyntax method,
        Compilation compilation)
    {
        var parameters = method.ParameterList.Parameters;
        foreach (var parameter in parameters)
        {
            var parameterSymbol = compilation.GetSymbol<IParameterSymbol>(parameter);
            var parameterTypeFullName = parameterSymbol.Type.ToDisplayString();
            
            yield return new ParameterSyntaxInfo
            {
                TypeFullName = parameterTypeFullName,
                Name = parameter.Identifier.Text,
            };
        }
    }

    public static T GetSymbol<T>(this Compilation compilation, ParameterSyntax parameter)
        where T : ISymbol
    {
        var model = compilation.GetSemanticModel(parameter.SyntaxTree);
        var symbol = (T)model.GetDeclaredSymbol(parameter)!;
        return symbol;
    }

    public static IEnumerable<AttributeData> QueryAttributeData(this AttributeListSyntax attributes, Compilation compilation)
    {
        var acceptedTrees = new HashSet<SyntaxTree>();
        foreach (var attribute in attributes.Attributes)
            acceptedTrees.Add(attribute.SyntaxTree);

        var parentSymbol = attributes.Parent!.GetDeclaredSymbol(compilation)!;
        var parentAttributes = parentSymbol.GetAttributes();
        
        foreach (var attribute in parentAttributes)
        {
            if (acceptedTrees.Contains(attribute.ApplicationSyntaxReference!.SyntaxTree))
            {
                yield return attribute;
            }
        }
    }

    public static IEnumerable<AttributeData> QueryAttributeData(this SyntaxList<AttributeListSyntax> attributeListSyntaxes,
        Compilation compilation)
    {
        return attributeListSyntaxes
            .SelectMany(x => x.QueryAttributeData(compilation));
    }

    public static IEnumerable<PropertySyntaxInfo> QueryPropertiesSyntaxInfo(this SyntaxNode typeSyntaxNode,
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
                    Accessibility = ToAccessibility(property.DeclaredAccessibility),
                    TypeFullName = property.Type.ToDisplayString(),
                    Name = property.Name,
                    HasGetter = property.GetMethod != null,
                    HasSetter = property.SetMethod != null,
                };
            }
        }
    }

    private static AccessibilityInfo ToAccessibility(Accessibility declaredAccessibility)
    {
        if (declaredAccessibility == Accessibility.Private)
        {
            return AccessibilityInfo.Private;
        }

        if (declaredAccessibility == Accessibility.Protected)
        {
            return AccessibilityInfo.Protected;
        }

        if (declaredAccessibility == Accessibility.Internal)
        {
            return AccessibilityInfo.Internal;
        }

        return AccessibilityInfo.Public;
    }
}