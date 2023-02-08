using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using T1.SourceGenerator.Attributes;
using T1.SourceGenerator.Utils;
using IndentStringBuilder = T1.SourceGenerator.Utils.IndentStringBuilder;

namespace T1.SourceGenerator.SourceGeneratorGen;

[Generator]
public class RoslynSourceGenerator : ISourceGenerator
{
	private readonly EmbeddedSources _embedded = new EmbeddedSources(typeof(RoslynSourceGenerator).Assembly);

	public void Initialize(GeneratorInitializationContext context)
	{
		context.RegisterForPostInitialization(i =>
		{
			i.AddSource("RoslynSourceGeneratorAttribute.g.cs", _embedded.LoadTemplateForEmitting("RoslynSourceGeneratorAttribute"));
			i.AddSource("TypeSyntaxInfo.g.cs", _embedded.LoadTemplateForEmitting("TypeSyntaxInfo"));
		});
	}

	public void Execute(GeneratorExecutionContext context)
	{
		var compilation = context.Compilation;
		var sourceGenerateExecutor = new RoslynSourceGenerateExecutor();
		var generateContext = new RoslynGeneratorExecutionContext(context)
        {
            AllTypes = compilation.GetAllTypes(),
			AllEnums = compilation.GetAllEnums()
        };

        var sourceGeneratorTypeList = QueryRoslynSourceGeneratorTypeList(generateContext.AllTypes)
			 .ToList();
		foreach (var sourceGeneratorType in sourceGeneratorTypeList)
		{
			var method = sourceGeneratorType.Methods.FirstOrDefault(x => x.Name == nameof(IRoslynSourceGenerator.Execute));
			if (method != null)
			{
				var methodSource = method.BodySourceCode;
				sourceGenerateExecutor.Execute(sourceGeneratorType, methodSource, generateContext);
			}
		}
	}

	private static IEnumerable<TypeSyntaxInfo> QueryRoslynSourceGeneratorTypeList(
		 List<TypeSyntaxInfo> typeSyntaxList)
	{
		foreach (var typeSyntaxInfo in typeSyntaxList)
		{
			if (typeSyntaxInfo.BaseTypes.All(x => x != typeof(IRoslynSourceGenerator).FullName))
			{
				continue;
			}

			var hasRoslynSourceGeneratorAttr = typeSyntaxInfo.Attributes
				 .Any(x => x.TypeFullName == typeof(RoslynSourceGeneratorAttribute).FullName);
			if (hasRoslynSourceGeneratorAttr)
			{
				yield return typeSyntaxInfo;
			}
		}
	}
}