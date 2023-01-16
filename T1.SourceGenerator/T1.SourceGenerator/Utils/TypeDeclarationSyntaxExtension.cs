using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace T1.SourceGenerator.Utils;


public static class TypeDeclarationSyntaxExtension
{
	const char NESTED_CLASS_DELIMITER = '+';
	const char NAMESPACE_CLASS_DELIMITER = '.';
	const char TYPEPARAMETER_CLASS_DELIMITER = '`';
	
	public static string GetFullName(this TypeDeclarationSyntax source)
	{
		if (source is null)
			throw new ArgumentNullException(nameof(source));

		var namespaces = new LinkedList<BaseNamespaceDeclarationSyntax>();
		var types = new LinkedList<TypeDeclarationSyntax>();
		for (var parent = source.Parent; parent is object; parent = parent.Parent)
		{
			if (parent is BaseNamespaceDeclarationSyntax namespaceSyntax)
			{
				namespaces.AddFirst(namespaceSyntax);
			}
			else if (parent is TypeDeclarationSyntax type)
			{
				types.AddFirst(type);
			}
		}

		var result = new StringBuilder();
		for (var item = namespaces.First; item is object; item = item.Next)
		{
			result.Append(item.Value.Name).Append(NAMESPACE_CLASS_DELIMITER);
		}
		for (var item = types.First; item is object; item = item.Next)
		{
			var type = item.Value;
			AppendName(result, type);
			result.Append(NESTED_CLASS_DELIMITER);
		}
		AppendName(result, source);

		return result.ToString();
	}

	static void AppendName(StringBuilder builder, TypeDeclarationSyntax type)
	{
		builder.Append(type.Identifier.Text);
		var typeArguments = type.TypeParameterList?.ChildNodes()
			 .Count(node => node is TypeParameterSyntax) ?? 0;
		if (typeArguments != 0)
			builder.Append(TYPEPARAMETER_CLASS_DELIMITER).Append(typeArguments);
	}
	
	public static ISymbol? GetDeclaredSymbol(this SyntaxNode node, Compilation compilation)
	{
		var model = compilation.GetSemanticModel(node.SyntaxTree);
		return model.GetDeclaredSymbol(node);
	}

	public static IEnumerable<AttributeDataSyntaxInfo> GetAttributesSyntaxInfo(this TypeDeclarationSyntax typeSyntax, Compilation compilation)
	{
		return typeSyntax.GetAttributeDataList(compilation)
			.Select(attr =>
				new AttributeDataSyntaxInfo
				{
					TypeFullName = attr.AttributeClass!.ToDisplayString(),
					ConstructorArguments = attr.ConstructorArguments.Select(constructorArgument =>
						new ArgumentDataSyntaxInfo
						{
							TypeFullName = constructorArgument.Type!.Name,
							ValueTypeFullName = constructorArgument.Value!.ToString(),
						}
					).ToList()
				});
	}
}