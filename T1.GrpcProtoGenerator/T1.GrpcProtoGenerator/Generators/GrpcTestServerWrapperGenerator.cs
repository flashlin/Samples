using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using T1.GrpcProtoGenerator.Common;
using T1.GrpcProtoGenerator.Generators.Models;

namespace T1.GrpcProtoGenerator.Generators
{
    [Generator]
    public class GrpcTestServerWrapperGenerator : IIncrementalGenerator
    {
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var protoFiles = context.AdditionalTextsProvider
            .Where(f => f.Path.EndsWith(".proto"));

        var protoFilesWithContent = protoFiles.Select((text, _) => new ProtoFileInfo
        {
            Path = text.Path,
            Content = text.GetText()!.ToString()
        });
        
        // Collect all proto files and process them together to handle imports
        var allProtoFiles = protoFilesWithContent.Collect();
        
        // Get compilation provider to check for package references
        var compilation = context.CompilationProvider;
        
        // Combine proto files with compilation information
        var protoFilesWithCompilation = allProtoFiles.Combine(compilation);

        context.RegisterSourceOutput(protoFilesWithCompilation, (spc, data) =>
        {
            var (allProtos, compilation) = data;
            
            // Only generate test code if NSubstitute package is available
            if (!IsNSubstitutePackageAvailable(compilation))
            {
                return;
            }
            
            var logger = InitializeLogger(spc);
            logger.LogWarning($"Starting test source generation for {allProtos.Length} proto files");
            
            var combinedModel = new ProtoModelResolver().CreateCombinedModel(allProtos);
            
            GenerateTestServiceFiles(spc, allProtos, combinedModel, logger, compilation);

            logger.LogInfo("Test source generation completed successfully");
        });
    }

        /// <summary>
        /// Initialize logger for source generation
        /// </summary>
        private ISourceGeneratorLogger InitializeLogger(SourceProductionContext spc)
        {
            return new SourceGeneratorLogger(spc.ReportDiagnostic, nameof(GrpcTestServerWrapperGenerator));
        }

        /// <summary>
        /// Generate test service files for all proto files
        /// </summary>
        private void GenerateTestServiceFiles(SourceProductionContext spc, ImmutableArray<ProtoFileInfo> allProtos, 
            ProtoModel combinedModel, ISourceGeneratorLogger logger, Compilation compilation)
        {
            logger.LogDebug($"Generating test service files for {allProtos.Length} proto files");
            
            foreach (var protoInfo in allProtos)
            {
                var model = ProtoParser.ParseProtoText(protoInfo.Content, protoInfo.Path);
                var protoFileName = protoInfo.GetProtoFileName();
                
                logger.LogDebug($"Generating test server files for {protoFileName}");
                
                // Generate test server files per proto file
                AddGeneratedSourceFile(spc, GenerateTestServerSource(model, combinedModel, compilation), 
                    $"Generated_{protoFileName}_test_server.cs");
            }
        }

        private static void AddGeneratedSourceFile(SourceProductionContext spc, string messagesSource, string sourceFileName)
        {
            if (string.IsNullOrEmpty(messagesSource))
            {
                return;
            }
            spc.AddSource(sourceFileName, SourceText.From(messagesSource, Encoding.UTF8));
        }

        private string GenerateTestServerSource(ProtoModel model, ProtoModel combineModel, Compilation compilation)
        {
            if (!ValidateTestServerSourceGeneration(model))
            {
                return string.Empty;
            }
            
            var sb = new IndentStringBuilder();
            sb.WriteLine("/// Auto-generated test code. Do not modify manually. Generated at " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
            
            // Setup using statements
            SetupTestServerSourceUsingStatements(sb, combineModel, compilation);
            
            // Group and generate services by namespace
            var servicesByNamespace = GroupServicesByNamespace(model);
            GenerateTestNamespaceBlocks(sb, servicesByNamespace, combineModel, compilation);

            return sb.ToString();
        }

        /// <summary>
        /// Validate if test server source generation is needed
        /// </summary>
        private bool ValidateTestServerSourceGeneration(ProtoModel model)
        {
            return model.Services.Any();
        }

        /// <summary>
        /// Setup using statements for test server source generation
        /// </summary>
        private void SetupTestServerSourceUsingStatements(IndentStringBuilder sb, ProtoModel combineModel, Compilation compilation)
        {
            var importNamespaces = CollectImportNamespacesForTestServer(combineModel);
            AddTestServerSpecificNamespaces(importNamespaces, compilation);
            GenerateUsingStatements(sb, importNamespaces);
        }

        /// <summary>
        /// Add test server-specific namespaces to the import list
        /// </summary>
        private void AddTestServerSpecificNamespaces(HashSet<string> importNamespaces, Compilation compilation)
        {
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            
            // Add NSubstitute namespace if the package is available
            if (IsNSubstitutePackageAvailable(compilation))
            {
                importNamespaces.Add("NSubstitute");
            }
        }

        /// <summary>
        /// Group services by their C# namespace
        /// </summary>
        private List<IGrouping<string, ProtoService>> GroupServicesByNamespace(ProtoModel model)
        {
            return model.Services
                .GroupBy(svc => svc.CsharpNamespace)
                .ToList();
        }

        /// <summary>
        /// Generate test namespace blocks with their contained services
        /// </summary>
        private void GenerateTestNamespaceBlocks(IndentStringBuilder sb, 
            List<IGrouping<string, ProtoService>> servicesByNamespace, 
            ProtoModel combineModel, Compilation compilation)
        {
            foreach (var namespaceGroup in servicesByNamespace)
            {
                GenerateTestNamespaceDeclaration(sb, namespaceGroup.Key);
                GenerateTestServicesInNamespace(sb, namespaceGroup, combineModel, compilation);
                GenerateTestNamespaceClosing(sb);
            }
        }

        /// <summary>
        /// Generate test namespace declaration
        /// </summary>
        private void GenerateTestNamespaceDeclaration(IndentStringBuilder sb, string namespaceValue)
        {
            sb.WriteLine($"namespace {namespaceValue}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all test services within a namespace
        /// </summary>
        private void GenerateTestServicesInNamespace(IndentStringBuilder sb, 
            IGrouping<string, ProtoService> namespaceGroup, 
            ProtoModel combineModel, Compilation compilation)
        {
            foreach (var svc in namespaceGroup)
            {
                GenerateMockServiceImplementation(sb, svc, combineModel, compilation);
            }
        }

        /// <summary>
        /// Generate test namespace closing brace
        /// </summary>
        private void GenerateTestNamespaceClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate mock service implementation class for a given service (if NSubstitute is available)
        /// </summary>
        private void GenerateMockServiceImplementation(IndentStringBuilder sb, ProtoService svc, ProtoModel combineModel, Compilation compilation)
        {
            // Check if NSubstitute package is available
            if (!IsNSubstitutePackageAvailable(compilation))
            {
                return;
            }

            var serviceInterface = $"I{svc.Name}GrpcService";
            var mockClass = $"NSubstituteFor{svc.Name}";
            
            sb.WriteLine($"public class {mockClass} : {serviceInterface}");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Generate private instance field
            sb.WriteLine($"private readonly {serviceInterface} _instance;");
            sb.WriteLine();
            
            // Generate constructor
            sb.WriteLine($"public {mockClass}()");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"_instance = Substitute.For<{serviceInterface}>();");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
            
            // Generate For property
            sb.WriteLine($"public {serviceInterface} For {{ get {{ return _instance; }} }}");
            sb.WriteLine();
            
            // Generate all interface methods that delegate to _instance
            foreach (var rpc in svc.Rpcs)
            {
                GenerateMockServiceMethod(sb, rpc, combineModel);
            }
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate a single mock service method that delegates to the substitute instance
        /// </summary>
        private void GenerateMockServiceMethod(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var methodSignature = CreateInterfaceMethodSignature(rpc, combineModel);
            
            // Generate method signature
            sb.WriteLine($"public {methodSignature.ReturnType} {rpc.Name}({methodSignature.Parameters})");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Generate method call delegation
            if (IsNullOrEmptyRequestType(rpc.RequestType))
            {
                sb.WriteLine($"return _instance.{rpc.Name}();");
            }
            else
            {
                sb.WriteLine($"return _instance.{rpc.Name}(request);");
            }
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Check if NSubstitute package is available in the project
        /// </summary>
        private bool IsNSubstitutePackageAvailable(Compilation compilation)
        {
            // Check if NSubstitute assembly is referenced in the compilation
            return compilation.ReferencedAssemblyNames
                .Any(assemblyName => assemblyName.Name.Equals("NSubstitute", StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Create method signature information for interface method
        /// </summary>
        private InterfaceMethodSignature CreateInterfaceMethodSignature(ProtoRpc rpc, ProtoModel messagesModel)
        {
            var rpcRequestType = messagesModel.FindCsharpTypeName(rpc.RequestType);
            var rpcResponseType = messagesModel.FindCsharpTypeName(rpc.ResponseType);

            return new InterfaceMethodSignature
            {
                Parameters = GenerateInterfaceMethodParameters(rpc, rpcRequestType),
                ReturnType = GenerateInterfaceMethodReturnType(rpc, rpcResponseType)
            };
        }

        /// <summary>
        /// Generate parameters for interface method
        /// </summary>
        private string GenerateInterfaceMethodParameters(ProtoRpc rpc, string rpcRequestType)
        {
            // Skip parameter if request type is "Null" or "google.protobuf.Empty"
            if (IsNullOrEmptyRequestType(rpc.RequestType))
            {
                return "";
            }
            
            return $"{rpcRequestType} request";
        }

        /// <summary>
        /// Generate return type for interface method
        /// </summary>
        private string GenerateInterfaceMethodReturnType(ProtoRpc rpc, string rpcResponseType)
        {
            // Use Task instead of Task<T> if response type is "Void" or "google.protobuf.Empty"
            if (IsVoidOrEmptyResponseType(rpc.ResponseType))
            {
                return "Task";
            }
            
            return $"Task<{rpcResponseType}>";
        }

        /// <summary>
        /// Check if the request type is considered null/empty (Null or google.protobuf.Empty)
        /// </summary>
        private static bool IsNullOrEmptyRequestType(string requestType)
        {
            return IsVoidOrNullMessageName(requestType);
        }

        /// <summary>
        /// Check if the response type is considered void/empty (Void or google.protobuf.Empty)
        /// </summary>
        private static bool IsVoidOrEmptyResponseType(string responseType)
        {
            return IsVoidOrNullMessageName(responseType);
        }

        /// <summary>
        /// Check if the message name should not generate a class (Void or Null)
        /// </summary>
        private static bool IsVoidOrNullMessageName(string messageName)
        {
            return messageName.Equals("Void", StringComparison.OrdinalIgnoreCase) ||
                   messageName.Equals("Null", StringComparison.OrdinalIgnoreCase) ||
                   messageName.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Collect namespaces from combined model messages and enums (for test server generation)
        /// </summary>
        private HashSet<string> CollectImportNamespacesForTestServer(ProtoModel combinedModel)
        {
            var namespaces = new HashSet<string>();
            
            // Collect namespaces from all messages in combined model
            foreach (var message in combinedModel.Messages)
            {
                namespaces.Add(message.CsharpNamespace);
            }
            
            // Collect namespaces from all enums in combined model
            foreach (var enumDef in combinedModel.Enums)
            {
                namespaces.Add(enumDef.CsharpNamespace);
            }
            
            return namespaces;
        }

        /// <summary>
        /// Generate using statements from the collected namespaces
        /// </summary>
        private void GenerateUsingStatements(IndentStringBuilder sb, HashSet<string> namespaces)
        {
            // Add default using statements
            sb.WriteLine("#nullable enable");
            sb.WriteLine("using System;");
            sb.WriteLine("using System.Collections.Generic;");
            
            // Add collected import namespaces (sorted for consistency)
            foreach (var ns in namespaces.OrderBy(x => x))
            {
                sb.WriteLine($"using {ns};");
            }
            
            sb.WriteLine();
        }
    }
}
