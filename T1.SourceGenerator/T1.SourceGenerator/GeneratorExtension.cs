using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator;

public static class GeneratorExtension
{
	public static string GetClassFullName(this GeneratorExecutionContext context, List<ClassDeclarationSyntax> allClasses, AttributeArgumentSyntax typeArgSyntax)
	{
		var className = typeArgSyntax.GetTypeofClassName();
		var classSyntax = allClasses.First(x => x.Identifier.ToString() == className);
		var classModel = context.Compilation.GetSemanticModel(classSyntax.SyntaxTree);
		var classNamedTypeSymbol = ModelExtensions.GetDeclaredSymbol(classModel, classSyntax)!;
		var classFullName = classNamedTypeSymbol.OriginalDefinition.ToString();
		return classFullName;
	}

	public static AttributeArgumentSyntax GetConstructorArgument(this AttributeSyntax attributeSyntax, int index)
	{
		return attributeSyntax.ArgumentList!.Arguments[index];
	}

	public static string GetTypeofClassName(this AttributeArgumentSyntax typeArgSyntax)
	{
		//var typeArgSyntaxExpr = typeArgSyntax.Expression.NormalizeWhitespace().ToFullString();
		//return GetContentInParentheses(typeArgSyntaxExpr);

		var typeOfSyntax = (typeArgSyntax.Expression as TypeOfExpressionSyntax)!;
		var typeSyntax = typeOfSyntax.Type;
		if (typeSyntax is IdentifierNameSyntax identifier)
		{
			return identifier.Identifier.ToString();
		}
		var baseTypeDeclarationSyntax = typeSyntax.Ancestors().OfType<BaseTypeDeclarationSyntax>().FirstOrDefault();
		return baseTypeDeclarationSyntax.GetNamespace() + "." + baseTypeDeclarationSyntax.Identifier.Text;
	}

	public static string GetNamespace(this TypeDeclarationSyntax type)
	{
		var namespaceNode = type.Ancestors()
			.OfType<NamespaceDeclarationSyntax>()
			.FirstOrDefault();
		if (namespaceNode != null)
		{
			return namespaceNode.Name.ToString();
		}
		return string.Empty;
	}


	public static string GetNamespace(this SyntaxNode attribute)
	{
		var parent = attribute.Parent;
		if (parent == null)
		{
			return string.Empty;
		}
		if (parent is ClassDeclarationSyntax classDecl)
		{
			return classDecl.GetNamespace();
		}
		else if (parent is NamespaceDeclarationSyntax namespaceDecl)
		{
			return namespaceDecl.Name.ToString();
		}

		if (parent is CompilationUnitSyntax compilationUnitSyntax)
		{
			var namespaceDecl = compilationUnitSyntax.Members
				.Where(x => x is NamespaceDeclarationSyntax)
				.Cast<NamespaceDeclarationSyntax>()
				.FirstOrDefault();
			return namespaceDecl.Name.ToString();
		}

		return GetNamespace(parent);
	}

	public static SyntaxNode GetRoot(this SyntaxNode node)
	{
		if (node.Parent == null)
		{
			return node;
		}

		if (node.Parent is CompilationUnitSyntax compilationUnitSyntax)
		{
			var namespaceDecl = compilationUnitSyntax.Members
				.Where(x => x is NamespaceDeclarationSyntax)
				.Cast<NamespaceDeclarationSyntax>()
				.FirstOrDefault();
			if (namespaceDecl != null)
			{
				return namespaceDecl;
			}

			return node.Parent;
		}

		return GetRoot(node.Parent);
	}

	private static string GetContentInParentheses(string value)
	{
		var match = Regex.Match(value, @"\(([^)]*)\)");
		return match.Groups[1].Value;
	}


	public static T? GetParentSyntax<T>(this SyntaxNode syntaxNode)
		where T : SyntaxNode
	{
		if (syntaxNode == null)
		{
			return default;
		}

		if (syntaxNode.Parent == null)
		{
			return default;
		}

		var parent = syntaxNode.Parent;
		if (parent.GetType() == typeof(T))
		{
			return parent as T;
		}

		return GetParentSyntax<T>(parent);
	}
}
