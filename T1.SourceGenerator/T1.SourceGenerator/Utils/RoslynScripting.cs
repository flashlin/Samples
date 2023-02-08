using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace T1.SourceGenerator.Utils;

public class RoslynScripting
{
    public RoslynScripting()
    {
        AddDefaultReferences();
    }

    public HashSet<PortableExecutableReference> References { get; set; } = new();

    public Either<Assembly, CompileError> Compile(string source)
    {
        var compilation = SetupCompilation(source);
        using var codeStream = new MemoryStream();
        // Actually compile the code
        var compilationResult = compilation.Emit(codeStream);

        // Compilation Error handling
        if (!compilationResult.Success)
        {
            var compileError = new CompileError();
            foreach (var diagnostic in compilationResult.Diagnostics)
            {
                compileError.Errors.Add(diagnostic.ToString());
            }

            return new Either<Assembly, CompileError>(compileError);
        }

        var assembly = Assembly.Load(codeStream.ToArray());
        //dynamic instance = assembly.CreateInstance("__ScriptExecution.__Executor")!;
        return new Either<Assembly, CompileError>(assembly);
    }

    public CSharpCompilation SetupCompilation(string source)
    {
        // Set up compilation Configuration
        var tree = SyntaxFactory.ParseSyntaxTree(source.Trim());
        var compilation = CSharpCompilation.Create("Executor.cs")
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: OptimizationLevel.Release))
            .WithReferences(References)
            .AddSyntaxTrees(tree);
        return compilation;
    }

    public bool AddAssembly(Type type)
    {
        try
        {
            if (References.Any(r => r.FilePath == type.Assembly.Location))
                return true;

            var systemReference = MetadataReference.CreateFromFile(type.Assembly.Location);
            References.Add(systemReference);
        }
        catch
        {
            return false;
        }

        return true;
    }

    public bool AddAssembly(string assemblyDll)
    {
        if (string.IsNullOrEmpty(assemblyDll)) return false;

        var file = Path.GetFullPath(assemblyDll);
        if (!File.Exists(file))
        {
            // check framework or dedicated runtime app folder
            var path = Path.GetDirectoryName(typeof(object).Assembly.Location)!;
            file = Path.Combine(path, assemblyDll);
            if (!File.Exists(file))
                return false;
        }

        if (References.Any(r => r.FilePath == file)) return true;

        try
        {
            var reference = MetadataReference.CreateFromFile(file);
            References.Add(reference);
        }
        catch
        {
            return false;
        }

        return true;
    }

    public void AddDefaultReferences()
    {
#if NETFRAMEWORK
	AddNetFrameworkDefaultReferences();
#else
        AddNetCoreDefaultReferences();
        // Core specific - not in base framework (for demonstration only)
        AddAssembly("System.Net.WebClient.dll");
#endif
    }

    public void AddNetFrameworkDefaultReferences()
    {
        AddAssembly("mscorlib.dll");
        AddAssembly("System.dll");
        AddAssembly("System.Core.dll");
        AddAssembly("Microsoft.CSharp.dll");
        AddAssembly("System.Net.Http.dll");
    }

    public void AddAssemblies(params string[] assemblies)
    {
        foreach (var file in assemblies)
            AddAssembly(file);
    }


    public void AddNetCoreDefaultReferences()
    {
        var rtPath = Path.GetDirectoryName(typeof(object).Assembly.Location) +
                     Path.DirectorySeparatorChar;

        AddAssembly(typeof(Enumerable));

        AddAssemblies(
            typeof(object).Assembly.Location,
            rtPath + "System.Private.CoreLib.dll",
            rtPath + "System.Runtime.dll",
            rtPath + "System.Console.dll",
            rtPath + "netstandard.dll",
            rtPath + "System.Text.RegularExpressions.dll", // IMPORTANT!
            rtPath + "System.Linq.dll",
            rtPath + "System.Linq.Expressions.dll", // IMPORTANT!
            rtPath + "System.IO.dll",
            rtPath + "System.Net.Primitives.dll",
            rtPath + "System.Net.Http.dll",
            rtPath + "System.Private.Uri.dll",
            rtPath + "System.Reflection.dll",
            rtPath + "System.ComponentModel.Primitives.dll",
            rtPath + "System.Globalization.dll",
            rtPath + "System.Collections.Concurrent.dll",
            rtPath + "System.Collections.NonGeneric.dll",
            rtPath + "Microsoft.CSharp.dll"
        );
    }
}