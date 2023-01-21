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

    internal List<TypeSyntaxInfo> AllTypes { get; set; } = new();

    public void Initialize(GeneratorInitializationContext context)
    {
        context.RegisterForPostInitialization(i =>
        {
            i.AddSource("WebApiClientConstructorInjectAttribute.g.cs", _embedded.LoadTemplateForEmitting("WebApiClientConstructorInjectAttribute"));
        });
    }

    public void Execute(GeneratorExecutionContext context)
    {
        var compilation = context.Compilation;
        AllTypes = compilation.GetAllTypes();

        foreach (var classType in QueryTypeWithWebApiClientAttribute(compilation))
        {
            var templateCode = _embedded.LoadTemplate("WebApiClient");
            var webApiClientAttrInfo =
                classType.Attributes.First(x => x.TypeFullName == typeof(WebApiClientAttribute).FullName);
            var webApiClientNamespaceArg =
                webApiClientAttrInfo.GetArgumentSyntaxInfo(nameof(WebApiClientAttribute.Namespace));
            var webApiClientNamespace = webApiClientNamespaceArg == null
                ? "T1.SourceGenerator"
                : webApiClientNamespaceArg.Value as string;

            templateCode = templateCode.Replace("T1.SourceGenerator", webApiClientNamespace);

            var webApiClientCtorAttrs = classType.Attributes.Where(x =>
                x.TypeFullName == typeof(WebApiClientConstructorInjectAttribute).FullName)
                .ToList();
            var webApiClientCtorInjectArguments = string.Empty;
            var webApiClientCtorInjectParamters = string.Empty;
            var webApiClientCtorInjectFields = string.Empty;
            if (webApiClientCtorAttrs.Any())
            {
                var ctorArguments = webApiClientCtorAttrs
                    .Select(x => GetConstructorInjectParameter(x.ConstructorArguments))
                    .ToList();
                webApiClientCtorInjectArguments = "," + string.Join(",", ctorArguments.Select(x => $"{x.TypeFullName} {x.Name}"));
                webApiClientCtorInjectParamters = string.Join("\t\n", ctorArguments.Select(x => $"_{x.Name} = {x.AssignCode};"));
                webApiClientCtorInjectFields = string.Join("\t\n", ctorArguments.Select(x => $"{x.TypeFullName} _{x.Name};"));
            }
            
            templateCode = templateCode.Replace("//<generate code: fields/>", webApiClientCtorInjectFields);
            templateCode = templateCode.Replace("//<generate code: ctor/>", webApiClientCtorInjectArguments);

            var initCtorCode = new IndentStringBuilder();
            initCtorCode.WriteLine(webApiClientCtorInjectParamters);
            templateCode = templateCode.Replace("//<generate code: initialize/>", initCtorCode.ToString());
            
            
            var webApiClientClassNameArg =
                webApiClientAttrInfo.GetArgumentSyntaxInfo(nameof(WebApiClientAttribute.ClientClassName));
            var webApiClientClassName = webApiClientClassNameArg == null ? $"{classType.TypeFullName.GetName()}Client" : webApiClientClassNameArg.Value as string;
            var methods = classType.Methods.Where(IsWebApiClientMethodAttributeAttached);
            var apiMethodCode = new IndentStringBuilder();
            foreach (var method in methods)
            {
                var methodName = method.Name;
                var webApiMethodAttr =
                    method.Attributes.First(x => x.TypeFullName == typeof(WebApiClientMethodAttribute).FullName);
                var apiPath =
                    webApiMethodAttr.ConstructorArguments[0].Value as string;

                var invokeMethodArg = webApiMethodAttr
                    .GetArgumentSyntaxInfo(nameof(WebApiClientMethodAttribute.Method));
                var invokeMethod = invokeMethodArg == null ? InvokeMethod.Post : (InvokeMethod)invokeMethodArg.Value!;
                var timeoutArg = webApiMethodAttr.GetArgumentSyntaxInfo(nameof(WebApiClientMethodAttribute.Timeout));
                var timeout = timeoutArg == null ? "00:00:30" : (string)timeoutArg.Value!;
                var methodArguments =
                    string.Join(",", method.Parameters.Select(x => $"{x.TypeFullName} {x.Name}"));
                var methodParameters =
                    string.Join(",", method.Parameters.Select(x => $"{x.Name}"));
                var methodReturnTypeFullName = method.ReturnTypeFullName;

                var webApiMethodContext = new WebApiMethodContext
                {
                    MethodReturnTypeFullName = methodReturnTypeFullName, 
                    MethodName = methodName, 
                    MethodArguments = methodArguments, 
                    MethodParameters = methodParameters, 
                    ApiPath = apiPath
                };
                if (invokeMethod == InvokeMethod.Post)
                {
                    WritePostMethod(webApiMethodContext, apiMethodCode, timeout);
                }
                else
                {
                    WriteGetMethod(webApiMethodContext, apiMethodCode, timeout);
                }
            }

            templateCode = templateCode.Replace("WebApiClient", webApiClientClassName);
            templateCode = templateCode.Replace("//<generate code: properties/>", apiMethodCode.ToString());
            context.AddSource($"{webApiClientClassName}.g.cs", SourceText.From(templateCode, Encoding.UTF8));
        }
    }

    private ConstructorInjectInfo GetConstructorInjectParameter(List<ArgumentSyntaxInfo> argumentSyntaxesInfo)
    {
        var typeFullName = argumentSyntaxesInfo[0].ValueTypeFullName;
        var name = (argumentSyntaxesInfo[1].Value as string)!;
        var assignCode = name;
        var assignCodeArg = argumentSyntaxesInfo.FirstOrDefault(x => x.Name == nameof(WebApiClientConstructorInjectAttribute.AssignCode));
        if (assignCodeArg != null)
        {
            assignCode = (assignCodeArg.Value as string)!;
        }
        return new ConstructorInjectInfo
        {
            TypeFullName = typeFullName,
            Name = name,
            AssignCode = assignCode,
        };
    }

    private static void WritePostMethod(WebApiMethodContext webApiMethodContext, IndentStringBuilder apiMethodCode,
        string timeout)
    {
        if (webApiMethodContext.MethodReturnTypeFullName != "void")
        {
            apiMethodCode.WriteLine($"public Task<{webApiMethodContext.MethodReturnTypeFullName}> {webApiMethodContext.MethodName}({webApiMethodContext.MethodArguments})");
            apiMethodCode.WriteLine("{");
            apiMethodCode.Indent++;
            if (string.IsNullOrEmpty(webApiMethodContext.MethodParameters))
            {
                webApiMethodContext.MethodParameters = "null";
            }
            apiMethodCode.WriteLine(
                $@"return PostDataAsync<{webApiMethodContext.MethodReturnTypeFullName}>(""{webApiMethodContext.ApiPath}"", {webApiMethodContext.MethodParameters}, TimeSpan.Parse(""{timeout}""));");
        }
        else
        {
            apiMethodCode.WriteLine($"public Task {webApiMethodContext.MethodName}({webApiMethodContext.MethodArguments})");
            apiMethodCode.WriteLine("{");
            apiMethodCode.Indent++;
            if (string.IsNullOrEmpty(webApiMethodContext.MethodParameters))
            {
                webApiMethodContext.MethodParameters = "null";
            }

            apiMethodCode.WriteLine($@"return PostVoidAsync(""{webApiMethodContext.ApiPath}"", {webApiMethodContext.MethodParameters}, TimeSpan.Parse(""{timeout}""));");
        }

        apiMethodCode.Indent--;
        apiMethodCode.WriteLine("}");
        apiMethodCode.WriteLine("");
    }
    

    private static void WriteGetMethod(WebApiMethodContext webApiMethodContext, IndentStringBuilder apiMethodCode,
        string timeout)
    {
        if (webApiMethodContext.MethodReturnTypeFullName != "void")
        {
            apiMethodCode.WriteLine($"public Task<{webApiMethodContext.MethodReturnTypeFullName}> {webApiMethodContext.MethodName}({webApiMethodContext.MethodArguments})");
            apiMethodCode.WriteLine("{");
            apiMethodCode.Indent++;
            if (string.IsNullOrEmpty(webApiMethodContext.MethodParameters))
            {
                webApiMethodContext.MethodParameters = "null";
            }

            apiMethodCode.WriteLine(
                $@"return GetDataAsync<{webApiMethodContext.MethodReturnTypeFullName}>(""{webApiMethodContext.ApiPath}"", {webApiMethodContext.MethodParameters}, TimeSpan.Parse(""{timeout}""));");
        }
        else
        {
            apiMethodCode.WriteLine($"public Task {webApiMethodContext.MethodName}({webApiMethodContext.MethodArguments})");
            apiMethodCode.WriteLine("{");
            apiMethodCode.Indent++;
            if (string.IsNullOrEmpty(webApiMethodContext.MethodParameters))
            {
                webApiMethodContext.MethodParameters = "null";
            }

            apiMethodCode.WriteLine($@"return GetVoidAsync(""{webApiMethodContext.ApiPath}"", {webApiMethodContext.MethodParameters}, TimeSpan.Parse(""{timeout}""));");
        }

        apiMethodCode.Indent--;
        apiMethodCode.WriteLine("}");
        apiMethodCode.WriteLine("");
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
}

public class ConstructorInjectInfo
{
    public string TypeFullName { get; set; } = null!;
    public string Name { get; set; } = null!;
    public string AssignCode { get; set; } = null!;
}