using System;
using Microsoft.CodeAnalysis;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis.CSharp;
using T1.Standard.DesignPatterns;

namespace T1.Roslyn
{
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

			AddAssembly(typeof(object));
			AddAssembly(typeof(Enumerable));
			AddAssemblies(new[]
				{
					"System.Private.CoreLib.dll",
					"System.Runtime.dll",
					"System.Console.dll",
					"netstandard.dll",
					"System.Text.RegularExpressions.dll",
					"System.Linq.dll",
					"System.Linq.Expressions.dll",
					"System.IO.dll",
					"System.Net.Primitives.dll",
					"System.Net.Http.dll",
					"System.Private.Uri.dll",
					"System.Reflection.dll",
					"System.ComponentModel.Primitives.dll",
					"System.Globalization.dll",
					"System.Collections.Concurrent.dll",
					"System.Collections.NonGeneric.dll",
					"Microsoft.CSharp.dll"
				}.Select(x => rtPath + x).ToArray()
			);
		}
	}
}