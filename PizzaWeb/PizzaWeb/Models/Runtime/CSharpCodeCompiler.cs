using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CSharp;
using System.CodeDom.Compiler;
using System.Reflection;

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
				syntaxTrees: new[] { syntaxTree },
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
			//var assembly = Assembly.Load(ms.ToArray());
//			return assembly;
			return ms.ToArray();
		}

		public Assembly Load(byte[] assemblyBytes)
		{
			var assembly = Assembly.Load(assemblyBytes);
			return assembly;
		}

		public void Execute(Assembly assembly)
		{
			var type = assembly.GetType("RoslynCompileSample.Writer");
			var obj = Activator.CreateInstance(type);
			type.InvokeMember("Write",
				 BindingFlags.Default | BindingFlags.InvokeMethod,
				 null,
				 obj,
				 new object[] { "Hello World" });
		}
	}
}
