using System.Text.Json;
using T1.SourceGenerator.Attributes;

namespace T1.SourceGenerator.Utils;

public class RoslynSourceGenerateExecutor
{
    private static readonly string[] DefaultUsingNamespaceList = new[]
    {
        "System",
        "System.Text",
        "System.Linq",
        "System.Linq.Expressions",
        "T1.SourceGenerator.Attributes",
    };

    public void Execute(TypeSyntaxInfo sourceGeneratorType, string generateMethodSource,
        RoslynGeneratorExecutionContext context)
    {
        var source = new IndentStringBuilder();
        foreach (var sNamespace in GetDistinctUsingNamespaceList(sourceGeneratorType))
        {
            source.WriteLine($"using {sNamespace};");
        }

        source.WriteLine("namespace T1.SourceGenerator.DynGenerators");
        source.WriteLine("{");

        source.Indent++;
        source.WriteLine("public class MySourceGenerator");
        source.WriteLine("{");
        source.Indent++;

        source.WriteLine("public void Generate(IRoslynGeneratorExecutionContext context)");
        source.WriteText(generateMethodSource);

        source.Indent--;
        source.WriteLine("}");

        source.Indent--;
        source.WriteLine("}");

        var roslyn = new RoslynScripting();
        roslyn.AddAssembly(typeof(IRoslynGeneratorExecutionContext));
        var result = roslyn.Compile(source.ToString());
        result.Match(assembly =>
        {
            dynamic instance = assembly.CreateInstance("T1.SourceGenerator.DynGenerators.MySourceGenerator")!;
            instance.Generate(context);
            return true;
        }, compilation => false);
    }

    private static List<string> GetDistinctUsingNamespaceList(TypeSyntaxInfo sourceGeneratorType)
    {
        var namespaceDict = new Dictionary<string, string>();
        foreach (var sNamespace in DefaultUsingNamespaceList)
        {
            namespaceDict[sNamespace] = sNamespace;
        }
        foreach (var sNamespace in sourceGeneratorType.UsingNamespaces)
        {
            namespaceDict[sNamespace] = sNamespace;
        }
        return namespaceDict.Keys.ToList();
    }
}