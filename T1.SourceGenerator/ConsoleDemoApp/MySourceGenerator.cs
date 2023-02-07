using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

public class MySourceGenerator : IRoslynSourceGenerator
{
    public void Execute(IRoslynGeneratorExecutionContext context)
    {
        context.AddSource("aaa.g.cs", "public enum AAA { BBB, CCC }");
    }
}
