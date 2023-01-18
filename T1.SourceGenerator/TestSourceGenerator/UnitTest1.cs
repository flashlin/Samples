using FluentAssertions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CodeAnalysis.Testing;
using T1.SourceGenerator.ApiClientGen;
using T1.SourceGenerator.Utils;

namespace TestSourceGenerator;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void test_method()
    {
        var compilation = GenerateCode(@"
namespace ConsoleDemoApp;
[WebApiClient(""JackProxy"")]
public interface IJackApi{
    [WebApiClientMethod(Method = InvokeMethod.Post, Timeout = ""1000"")]
    void Test(int a);
}");
        
        var allTypes = EmitAndGetAllTypes(compilation);
        
        allTypes[0].Methods[0].Name.Should()
            .Be("Test");
    }

    public CSharpCompilation GenerateCode(string sourceCode)
    {
        var compilation = CSharpCompilation.Create("MyCompilation")
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .AddSyntaxTrees(CSharpSyntaxTree.ParseText(sourceCode))
            .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location));
        return compilation;
    }

    public List<TypeSyntaxInfo> EmitAndGetAllTypes(CSharpCompilation compilation)
    {
        var generator = new ApiClientGenerator();
        var driver = CSharpGeneratorDriver.Create(generator);
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);
        return generator.AllTypes;
    }
}