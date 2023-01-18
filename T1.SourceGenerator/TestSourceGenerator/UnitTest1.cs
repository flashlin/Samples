using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CodeAnalysis.Testing;
using T1.SourceGenerator.ApiClientGen;

namespace TestSourceGenerator;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var compilation = GenerateCode(@"
[WebApiClient(""JackProxy"")]
public interface IJackApi{
    [WebApiClientMethod(Method = InvokeMethod.Post, Timeout = ""1000"")]
    void Test(int a);
}");
        
        Emit(compilation);
        Assert.Pass();
    }

    public CSharpCompilation GenerateCode(string sourceCode)
    {
        var compilation = CSharpCompilation.Create("MyCompilation")
            //.WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .WithOptions(new CSharpCompilationOptions(OutputKind.ConsoleApplication))
            .AddSyntaxTrees(CSharpSyntaxTree.ParseText(sourceCode))
            .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location));
        return compilation;
    }

    public void Emit(CSharpCompilation compilation)
    {
        var generator = new ApiClientGenerator();
        var driver = CSharpGeneratorDriver.Create(generator);
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var t = generator.Debug.ToList();
    }
}