using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

[RoslynSourceGenerator]
public class MySourceGenerator : IRoslynSourceGenerator
{
    public void Execute(IRoslynGeneratorExecutionContext context)
    {
        var typeList = context.AllTypes.Where( x=> x.TypeFullName == "aa").ToList();
        context.AddSource("aaa.g.cs", "namespace ConsoleDemoApp; public enum AAA { BBB, CCC }");
    }
}
