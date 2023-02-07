using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

[RoslynSourceGenerator]
public class MySourceGenerator : IRoslynSourceGenerator
{
    public void Execute(IRoslynGeneratorExecutionContext context)
    {
        context.AddSource("aaa.g.cs", "namespace ConsoleDemoApp; public enum AAA { BBB, CCC }");
    }
}
