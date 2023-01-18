using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using T1.SourceGenerator.Attributes;
using T1.SourceGenerator.AutoMappingGen;
using T1.SourceGenerator.Utils;

[assembly: InternalsVisibleTo("TestSourceGenerator")]
namespace T1.SourceGenerator.ApiClientGen;

[Generator]
public class ApiClientGenerator : ISourceGenerator
{
    private readonly EmbeddedSources _embedded = new EmbeddedSources(typeof(ApiClientGenerator).Assembly);

    internal List<TypeSyntaxInfo> AllTypes { get; set; }

    public void Initialize(GeneratorInitializationContext context)
    {
        // context.RegisterForPostInitialization(i =>
        // {
        //     i.AddSource("AutoMappingAttribute.g.cs", _embedded.LoadTemplateForEmitting("AutoMappingAttribute"));
        // });
    }

    public void Execute(GeneratorExecutionContext context)
    {
        var compilation = context.Compilation;
        AllTypes = compilation.GetAllTypes();

        var templateCode = _embedded.LoadTemplate("WebApiClient");
        foreach (var classType in QueryTypeWithWebApiClientAttribute(compilation))
        {
            var webApiClientAttrInfo = classType.Attributes.First(x => x.TypeFullName == typeof(WebApiClientAttribute).FullName);
            var webApiClientClassName = webApiClientAttrInfo.ConstructorArguments[0].Value as string;
            var methods = classType.Methods.Where(IsWebApiClientMethodAttributeAttached);
            foreach (var method in methods)
            {
                var webApiMethodAttr = method.Attributes.First(x => x.TypeFullName == typeof(WebApiClientMethodAttribute).FullName);
                var invokeMethod = webApiMethodAttr.ConstructorArguments.First(x => x.Name == nameof(WebApiClientMethodAttribute.Method)).Value as string;
                var timeout = webApiMethodAttr.ConstructorArguments
                    .First(x => x.Name == nameof(WebApiClientMethodAttribute.Timeout)).Value as string;

                templateCode = templateCode.Replace("WebApiClient", webApiClientClassName);
                //webApiMethodAttr.ConstructorArguments.Where(x => x.TypeFullName);
                context.AddSource($"{webApiClientClassName}.g.cs", SourceText.From(templateCode, Encoding.UTF8));
            }
        }
    }

    private bool IsWebApiClientMethodAttributeAttached(MethodSyntaxInfo method)
    {
        return method.Attributes.Any(x => x.TypeFullName == typeof(WebApiClientMethodAttribute).FullName);
    }

    private static IEnumerable<TypeSyntaxInfo> QueryTypeWithWebApiClientAttribute(Compilation compilation)
    {
        foreach (var type in compilation.GetAllTypes())
        {
            var webApiAttr = type.SyntaxNode.AttributeLists.QueryAttributesSyntaxInfo(compilation)
                .FirstOrDefault(x => x.TypeFullName == typeof(WebApiClientAttribute).FullName);
            if (webApiAttr != null)
            {
                yield return type;
            } 
        }
    }


    private static IEnumerable<WebApiDeclarationInfo> GetWebApiAttributes(TypeSyntaxInfo type,
        Compilation compilation)
    {
        var allTypes = compilation.GetAllTypes();
        var attributeDataSyntaxInfos = type.SyntaxNode.AttributeLists.QueryAttributesSyntaxInfo(compilation)
            .Where(x => x.TypeFullName == typeof(WebApiClientAttribute).FullName)
            .ToList();
        foreach (var x in attributeDataSyntaxInfos)
        {
            var methods = type.Methods;
            foreach (var method in methods)
            {
                yield return new WebApiDeclarationInfo
                {
                    HttpClientClassName = x.ConstructorArguments[0].Value as string,
                    Method = method
                };
            }
        }
    }
}

public class WebApiDeclarationInfo
{
    public string? HttpClientClassName { get; set; }
    public MethodSyntaxInfo Method { get; set; } = null!;
}