namespace T1.SourceGenerator.Attributes;

[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
public class RoslynSourceGeneratorAttribute : Attribute
{
}

public interface IRoslynGeneratorExecutionContext
{
    public List<TypeSyntaxInfo> AllTypes { get; }
    public void AddSource(string name, string sourceText);
}

public interface IRoslynSourceGenerator
{
    public void Execute(IRoslynGeneratorExecutionContext context);
}