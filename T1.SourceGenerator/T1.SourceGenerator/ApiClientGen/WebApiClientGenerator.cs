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
public class WebApiClientGenerator : ISourceGenerator
{
    private readonly EmbeddedSources _embedded = new EmbeddedSources(typeof(WebApiClientGenerator).Assembly);

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
            var webApiClientAttrInfo =
                classType.Attributes.First(x => x.TypeFullName == typeof(WebApiClientAttribute).FullName);
            var webApiClientClassName = webApiClientAttrInfo.ConstructorArguments[0].Value as string;
            var methods = classType.Methods.Where(IsWebApiClientMethodAttributeAttached);
            var apiMethodCode = new IndentStringBuilder();
            foreach (var method in methods)
            {
                var methodName = method.Name;
                var webApiMethodAttr =
                    method.Attributes.First(x => x.TypeFullName == typeof(WebApiClientMethodAttribute).FullName);
                var apiPath =
                    webApiMethodAttr.ConstructorArguments[0].Value as string;
                var invokeMethod = (InvokeMethod)webApiMethodAttr
                    .GetArgumentSyntaxInfo(nameof(WebApiClientMethodAttribute.Method)).Value!;
                var timeout =
                    webApiMethodAttr.GetArgumentSyntaxInfo(nameof(WebApiClientMethodAttribute.Timeout)).Value as string;
                var methodArguments =
                    string.Join(",", method.Parameters.Select(x => $"{x.TypeFullName} {x.Name}"));
                var methodParameters =
                    string.Join(",", method.Parameters.Select(x => $"{x.Name}"));
                var methodReturnTypeFullName = method.ReturnTypeFullName;

                if (methodReturnTypeFullName != "void")
                {
                    apiMethodCode.WriteLine($"public Task<{methodReturnTypeFullName}> {methodName}({methodArguments})");
                    apiMethodCode.WriteLine("{");
                    apiMethodCode.Indent++;
                    apiMethodCode.WriteLine($@"return PostDataAsync<{methodReturnTypeFullName}>(""{apiPath}"", {methodParameters});");
                }
                else
                {
                    apiMethodCode.WriteLine($"public Task {methodName}({methodArguments})");
                    apiMethodCode.WriteLine("{");
                    apiMethodCode.Indent++;
                    apiMethodCode.WriteLine($@"return PostVoidAsync(""{apiPath}"", {methodParameters});");
                }

                apiMethodCode.Indent--;
                apiMethodCode.WriteLine("}");
                apiMethodCode.WriteLine("");
            }

            templateCode = templateCode.Replace("WebApiClient", webApiClientClassName);
            templateCode = templateCode.Replace("//<generate code: properties/>", apiMethodCode.ToString());
            context.AddSource($"{webApiClientClassName}.g.cs", SourceText.From(templateCode, Encoding.UTF8));
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