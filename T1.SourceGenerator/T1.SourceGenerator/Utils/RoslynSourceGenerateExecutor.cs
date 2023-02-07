using T1.SourceGenerator.Attributes;

namespace T1.SourceGenerator.Utils;

public class RoslynSourceGenerateExecutor
{
    public void Execute(string generateMethodSource, RoslynGeneratorExecutionContext context)
    {
        var source = new IndentStringBuilder();
        source.WriteLine("using System;");
        source.WriteLine("namespace T1.SourceGenerator.DynGenerators");
        source.WriteLine("{");

        source.Indent++;
        source.WriteLine("public class MySourceGenerator");
        source.WriteLine("{");
        source.Indent++;

        source.WriteLine("public void Generate(RoslynGeneratorExecutionContext context) {");
        source.Indent++;
        source.WriteText(generateMethodSource);
        source.Indent--;
        source.WriteLine("}");

        source.Indent--;
        source.WriteLine("}");

        source.Indent--;
        source.WriteLine("}");

        var roslyn = new RoslynScripting();
        var result = roslyn.Compile(source.ToString());
        result.Match(assembly =>
        {
            dynamic instance = assembly.CreateInstance("T1.SourceGenerator.DynGenerators.MySourceGenerator")!;
            instance.Generate(context);
            return true;
        }, compilation => false);

    }
}