using T1.SourceGenerator.Attributes;

namespace T1.SourceGenerator.Utils;

public class RoslynSourceGenerateExecutor
{
    public void Execute(string generateMethodSource, RoslynGeneratorExecutionContext context)
    {
        var source = new IndentStringBuilder();
        source.WriteLine("using System;");
        source.WriteLine("using System.Text;");
        source.WriteLine("using T1.SourceGenerator.Attributes;");
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
}