using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CSharp;
using System.CodeDom.Compiler;

namespace PizzaWeb.Models.Runtime
{
    //Microsoft.CodeAnalysis.CSharp
    public class CSharpCodeCompiler
    {
        public byte[] Compile(string code)
        {
            var syntaxTree = CSharpSyntaxTree.ParseText(code);

            var assemblyName = Path.GetRandomFileName();
            var references = new MetadataReference[]
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location)
            };

            var compilation = CSharpCompilation.Create(
                assemblyName,
                syntaxTrees: new[] {syntaxTree},
                references: references,
                options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

            using var ms = new MemoryStream();
            var result = compilation.Emit(ms);
            if (!result.Success)
            {
                var failures = result.Diagnostics.Where(diagnostic =>
                    diagnostic.IsWarningAsError ||
                    diagnostic.Severity == DiagnosticSeverity.Error);
                throw new CompileException(failures);
            }
            ms.Seek(0, SeekOrigin.Begin);
            return ms.ToArray();
        }
    }
}