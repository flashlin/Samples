using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using T1.SourceGenerator.Attributes;
using T1.SourceGenerator.AutoMappingGen;

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
                UsingNamespaces = x.QueryUsingNamespaceList().ToList(),
                Attributes = x.AttributeLists.QueryAttributesSyntaxInfo(compilation).ToList(),
                TypeFullName = x.GetFullName(),
                BaseTypes = x.QueryBaseTypeInfo(compilation).ToList(),
                Methods = x.QueryMethodsSyntaxInfo(compilation).ToList(),
                Properties = x.QueryPropertiesSyntaxInfo(compilation).ToList(),
                Fields = x.QueryFieldsSyntaxInfo(compilation).ToList(),
            })
            .ToList();
    }

    public static IEnumerable<AttributeSyntaxInfo> GetAttributeDeclarations<TAttribute>(this Compilation compilation,
        TypeSyntaxInfo type)
    {
        return type.Attributes
            .Where(x => x.TypeFullName == typeof(TAttribute).FullName);
    }

    public static List<EnumSyntaxInfo> GetAllEnums(this Compilation compilation)
    {
        return compilation.SyntaxTrees
            .SelectMany(syntaxTree => syntaxTree.GetRoot().DescendantNodes().OfType<EnumDeclarationSyntax>())
            .Select(enumType => new EnumSyntaxInfo
            {
                Name = enumType.Identifier.Text,
                Values = enumType.Members.Select(member => member.GetEnumMemberInfo()).ToList(),
            })
            .ToList();
    }

    private static EnumMemberSyntaxInfo GetEnumMemberInfo(this EnumMemberDeclarationSyntax member)
    {
        return new EnumMemberSyntaxInfo
        {
            Name = member.Identifier.Text,
            Value = member.GetEnumValue()
        };
    }

    private static string GetEnumValue(this EnumMemberDeclarationSyntax member)
    {
        if (member.EqualsValue == null)
        {
            return string.Empty;
        }

        var valueNode = member.EqualsValue.Value;
        var value = ((LiteralExpressionSyntax) valueNode).Token.Value!;
        return $"{value}";
    }


    public static string GetName(this string fullname)
    {
        var idx = fullname.LastIndexOf(".", StringComparison.Ordinal);
        if (idx == -1)
        {
            return fullname;
        }

        return fullname.Substring(idx + 1);
    }

    public static IEnumerable<string> QueryBaseTypeInfo(this TypeDeclarationSyntax typeDeclaration,
        Compilation compilation)
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

    private static IEnumerable<UsingDirectiveSyntax> QueryUsingDirectiveList(this TypeDeclarationSyntax node)
    {
        var parentNode = node.Parent;
        while (!(parentNode is CompilationUnitSyntax))
        {
            parentNode = parentNode!.Parent;
        }

        var compilationUnit = (CompilationUnitSyntax) parentNode;
        return compilationUnit.Usings;
    }

    public static IEnumerable<string> QueryUsingNamespaceList(this TypeDeclarationSyntax node)
    {
        return node.QueryUsingDirectiveList()
            .Select(x => x.Name.ToString())
            .ToList();
    }

    public static IEnumerable<MethodSyntaxInfo> QueryMethodsSyntaxInfo(this TypeDeclarationSyntax typeDeclaration,
        Compilation compilation)
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

    public static IEnumerable<AttributeSyntaxInfo> QueryAttributesSyntaxInfo(
        this MethodDeclarationSyntax methodDeclarationSyntax, Compilation compilation)
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
        var symbol = (T) model.GetDeclaredSymbol(parameter)!;
        return symbol;
    }

    public static IEnumerable<AttributeData> QueryAttributeData(this AttributeListSyntax attributes,
        Compilation compilation)
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

    public static IEnumerable<AttributeData> QueryAttributeData(
        this SyntaxList<AttributeListSyntax> attributeListSyntaxes,
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
                var property = (IPropertySymbol) member;
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

    public static IEnumerable<FieldSyntaxInfo> QueryFieldsSyntaxInfo(this SyntaxNode typeSyntaxNode,
        Compilation compilation)
    {
        var model = compilation.GetSemanticModel(typeSyntaxNode.SyntaxTree);
        var symbol = (model.GetDeclaredSymbol(typeSyntaxNode) as INamedTypeSymbol)!;
        foreach (var member in symbol.GetMembers())
        {
            if (member.Kind == SymbolKind.Field)
            {
                var fieldSymbol = (IFieldSymbol) member;
                var attributeSymbols = fieldSymbol.GetAttributes()
                    .Select(x => x.ToAttributeSyntaxInfo());

                // 检查字段是否具有初始化代码
                var hasInitialization = fieldSymbol.DeclaringSyntaxReferences.Any(syntaxRef =>
                {
                    var syntaxNode = syntaxRef.GetSyntax();
                    return syntaxNode is VariableDeclaratorSyntax {Initializer: not null};
                });

                yield return new FieldSyntaxInfo
                {
                    Accessibility = ToAccessibility(fieldSymbol.DeclaredAccessibility),
                    TypeFullName = fieldSymbol.Type.ToDisplayString(),
                    Name = fieldSymbol.Name,
                    IsReadOnly = fieldSymbol.IsReadOnly,
                    Attributes = attributeSymbols.ToList(),
                    HasInitialization = hasInitialization,
                    InitializationCode = hasInitialization ? GetInitializationCode(fieldSymbol) : string.Empty,
                };
            }
        }
    }

    private static string GetInitializationCode(IFieldSymbol fieldSymbol)
    {
        var syntaxNode = fieldSymbol.DeclaringSyntaxReferences[0].GetSyntax();
        if (syntaxNode is VariableDeclaratorSyntax variableDeclarator &&
            variableDeclarator.Initializer != null)
        {
            return variableDeclarator.Initializer.Value.ToFullString();
        }

        if (syntaxNode is EventFieldDeclarationSyntax eventFieldDeclaration &&
            eventFieldDeclaration.Declaration.Variables.Count > 0 &&
            eventFieldDeclaration.Declaration.Variables[0].Initializer != null)
        {
            return eventFieldDeclaration.Declaration.Variables[0].Initializer.Value.ToFullString();
        }

        return string.Empty;
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