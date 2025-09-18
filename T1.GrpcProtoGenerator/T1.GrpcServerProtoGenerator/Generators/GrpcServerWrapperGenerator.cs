using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;

namespace T1.GrpcProtoGenerator.Generators
{
    [Generator]
    public class GrpcServerWrapperGenerator : IIncrementalGenerator
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

            context.RegisterSourceOutput(allProtoFiles, (spc, allProtos) =>
            {
                // Create logger for this execution context
                ISourceGeneratorLogger logger = new SourceGeneratorLogger(spc.ReportDiagnostic, nameof(GrpcServerWrapperGenerator));
                
                logger.LogWarning($"Starting source generation for {allProtos.Length} proto files");
                
                // Generate a single combined messages file for all proto files
                var combinedModel = CreateCombinedModel(allProtos);
                AddGeneratedSourceFile(spc, GenerateWrapperGrpcMessageSource(combinedModel), "Generated_messages.cs");
                
                var protoResolver = new ProtoImportResolver(allProtos);
                foreach (var protoInfo in allProtos)
                {
                    var model = ProtoParser.ParseProtoText(protoInfo.Content);
                    var enrichedModel = protoResolver.EnrichModelWithImports(model, protoInfo.Path);
                    var protoFileName = protoInfo.GetProtoFileName();
                    
                    logger.LogDebug($"Generating server and client files for {protoFileName}");
                    // Generate server and client files per proto file
                    AddGeneratedSourceFile(spc, GenerateWrapperServerSource(enrichedModel, combinedModel), $"Generated_{protoFileName}_server.cs");
                    AddGeneratedSourceFile(spc, GenerateWrapperClientSource(enrichedModel, combinedModel, protoResolver, protoInfo.Path), $"Generated_{protoFileName}_client.cs");
                }
                
                logger.LogInfo("Source generation completed successfully");
            });
        }

        /// <summary>
        /// Create a combined model with unique messages and enums from all proto files
        /// </summary>
        private ProtoModel CreateCombinedModel(ImmutableArray<ProtoFileInfo> allProtos)
        {
            var combinedModel = new ProtoModel();
            
            // Collect all messages and enums from all proto files
            var allMessages = new List<ProtoMessage>();
            var allEnums = new List<ProtoEnum>();
            var allSvcs = new List<ProtoService>();
            
            foreach (var protoInfo in allProtos)
            {
                var model = ProtoParser.ParseProtoText(protoInfo.Content);
                allMessages.AddRange(model.Messages);
                allEnums.AddRange(model.Enums);
                allSvcs.AddRange(model.Services);
            }
            
            // Add unique messages (based on FullName or Name)
            var uniqueMessages = allMessages
                .GroupBy(m => m.GetFullName())
                .Select(g => g.First())
                .ToList();
            
            foreach (var message in uniqueMessages)
            {
                combinedModel.Messages.Add(message);
            }
            
            // Add unique enums (based on Name and Namespace)
            var uniqueEnums = allEnums
                .GroupBy(e => $"{e.GetFullName()}")
                .Select(g => g.First())
                .ToList();
            
            foreach (var enumDef in uniqueEnums)
            {
                combinedModel.Enums.Add(enumDef);
            }

            foreach (var svc in allSvcs)
            {
                foreach (var rpc in svc.Rpcs)
                {
                    rpc.RequestFullTypename = combinedModel.FindRpcFullTypename(rpc.RequestType);
                    rpc.ResponseFullTypename = combinedModel.FindRpcFullTypename(rpc.ResponseType);
                }
                combinedModel.Services.Add(svc);
            }
            
            return combinedModel;
        }

        private static void AddGeneratedSourceFile(SourceProductionContext spc, string messagesSource, string sourceFileName)
        {
            if (string.IsNullOrEmpty(messagesSource))
            {
                return;
            }
            spc.AddSource(sourceFileName, SourceText.From(messagesSource, Encoding.UTF8));
        }

        private string GenerateWrapperGrpcMessageSource(ProtoModel combineModel)
        {
            var sb = new StringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            // Generate message classes and enums grouped by namespace
            var allNamespaces = CollectAllUniqueNamespaces(combineModel.Messages, combineModel.Enums);
            foreach (var namespaceValue in allNamespaces)
            {
                sb.AppendLine($"namespace {namespaceValue}");
                sb.AppendLine("{");

                GenerateMessagesForNamespace(sb, combineModel.Messages, namespaceValue, combineModel);
                GenerateEnumsForNamespace(sb, combineModel.Enums, namespaceValue);

                sb.AppendLine("}");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generate basic using statements for message source files
        /// </summary>
        private void GenerateBasicUsingStatements(StringBuilder sb)
        {
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine();
        }

        /// <summary>
        /// Collect all unique namespaces from messages and enums
        /// </summary>
        private List<string> CollectAllUniqueNamespaces(List<ProtoMessage> modelMessages, List<ProtoEnum> modelEnums)
        {
            var messageNamespaces = modelMessages
                .Select(msg => msg.CsharpNamespace)
                .ToHashSet();
            
            var enumNamespaces = modelEnums
                .Select(e => e.CsharpNamespace)
                .ToHashSet();
            
            return messageNamespaces.Union(enumNamespaces).OrderBy(ns => ns).ToList();
        }

        /// <summary>
        /// Generate all message classes for a specific namespace
        /// </summary>
        private void GenerateMessagesForNamespace(StringBuilder sb, List<ProtoMessage> modelMessages,
            string namespaceValue, ProtoModel combineModel)
        {
            var messagesInNamespace = modelMessages
                .Where(msg => msg.CsharpNamespace == namespaceValue)
                .ToList();
            
            foreach (var msg in messagesInNamespace)
            {
                GenerateSingleMessageClass(sb, msg, combineModel);
            }
        }

        /// <summary>
        /// Generate all enum classes for a specific namespace
        /// </summary>
        private void GenerateEnumsForNamespace(StringBuilder sb, List<ProtoEnum> modelEnums, string namespaceValue)
        {
            var enumsInNamespace = modelEnums
                .Where(e => e.CsharpNamespace == namespaceValue)
                .ToList();
            
            foreach (var enumDef in enumsInNamespace)
            {
                GenerateSingleEnumClass(sb, enumDef);
            }
        }

        /// <summary>
        /// Generate a single message class with its properties
        /// </summary>
        private void GenerateSingleMessageClass(StringBuilder sb, ProtoMessage msg, ProtoModel combineModel)
        {
            sb.AppendLine($"    public class {msg.GetCsharpTypeName()}");
            sb.AppendLine("    {");
            
            foreach (var field in msg.Fields)
            {
                var baseType = MapProtoCTypeToCSharp(field.Type, combineModel);
                var csType = field.IsRepeated ? $"List<{baseType}>" : baseType;
                var propertyName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                sb.AppendLine($"        public {csType} {propertyName} {{ get; set; }}");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate a single enum class with its values
        /// </summary>
        private void GenerateSingleEnumClass(StringBuilder sb, ProtoEnum enumDef)
        {
            sb.AppendLine($"    public enum {enumDef.GetCsharpTypeName()}");
            sb.AppendLine("    {");
            
            foreach (var val in enumDef.Values)
            {
                sb.AppendLine($"        {val.Name} = {val.Value},");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        private string GenerateWrapperClientSource(ProtoModel model, ProtoModel combinedModel, ProtoImportResolver resolver, string protoPath)
        {
            if (!model.Services.Any())
            {
                return string.Empty;
            }
            
            var sb = new StringBuilder();
            
            // Collect import namespaces for messages and enums only
            var importNamespaces = CollectImportNamespacesForServer(combinedModel);
            
            // Add specific using statements for client generation
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            
            GenerateUsingStatements(sb, importNamespaces);
            
            // Group services by CsharpNamespace
            var servicesByNamespace = model.Services
                .GroupBy(svc => svc.CsharpNamespace)
                .ToList();

            // Generate namespace blocks for each group
            foreach (var namespaceGroup in servicesByNamespace)
            {
                var namespaceValue = namespaceGroup.Key;
                sb.AppendLine($"namespace {namespaceValue}");
                sb.AppendLine("{");

                // Generate all client interfaces and wrappers for this namespace
                foreach (var svc in namespaceGroup)
                {
                    GenerateClientInterface(sb, svc);
                    GenerateClientWrapper(sb, svc, model);
                }

                sb.AppendLine("}");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generate client interface for a given service
        /// </summary>
        private void GenerateClientInterface(StringBuilder sb, ProtoService svc)
        {
            var clientInterface = $"I{svc.Name}Client";
            sb.AppendLine($"    public interface {clientInterface}");
            sb.AppendLine("    {");
            
            foreach (var rpc in svc.Rpcs)
            {
                sb.AppendLine($"        Task<{rpc.ResponseType}GrpcDto> {rpc.Name}Async({rpc.RequestType}GrpcDto request, CancellationToken cancellationToken = default);");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate client wrapper class for a given service
        /// </summary>
        private void GenerateClientWrapper(StringBuilder sb, ProtoService svc, ProtoModel model)
        {
            var originalNamespace = svc.CsharpNamespace;
            var clientInterface = $"I{svc.Name}Client";
            var wrapper = $"{svc.Name}ClientWrapper";
            var grpcClient = $"{originalNamespace}.{svc.Name}.{svc.Name}Client";
            
            sb.AppendLine($"    public class {wrapper} : {clientInterface}");
            sb.AppendLine("    {");
            sb.AppendLine($"        private readonly {grpcClient} _inner;");
            sb.AppendLine($"        public {wrapper}({grpcClient} inner) {{ _inner = inner; }}");
            sb.AppendLine();

            foreach (var rpc in svc.Rpcs)
            {
                GenerateClientMethod(sb, rpc, model, originalNamespace);
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate a single client method implementation
        /// </summary>
        private void GenerateClientMethod(StringBuilder sb, ProtoRpc rpc, ProtoModel model, string originalNamespace)
        {
            // Determine the correct namespace for request and response types
            sb.AppendLine($"        public async Task<{rpc.ResponseType}GrpcDto> {rpc.Name}Async({rpc.RequestType}GrpcDto request, CancellationToken cancellationToken = default)");
            sb.AppendLine("        {");
            sb.AppendLine($"            var grpcReq = new {originalNamespace}.{rpc.RequestType}();");
            
            // Map request fields
            var requestMessage = model.FindMessage(rpc.RequestType);
            if (requestMessage != null)
            {
                foreach (var field in requestMessage.Fields)
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    sb.AppendLine($"            grpcReq.{propName} = request.{propName};");
                }
            }
            else
            {
                // Handle external types
            }
            
            sb.AppendLine($"            var grpcResp = await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
            sb.AppendLine($"            var dto = new {rpc.ResponseType}GrpcDto();");
            
            // Map response fields
            var responseMessage = model.FindMessage(rpc.ResponseType);
            if (responseMessage != null)
            {
                foreach (var field in responseMessage.Fields)
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    sb.AppendLine($"            dto.{propName} = grpcResp.{propName};");
                }
            }
            else
            {
                // Handle external types
            }
            
            sb.AppendLine("            return dto;");
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        private string GenerateWrapperServerSource(ProtoModel model, 
            ProtoModel combineModel)
        {
            if (!model.Services.Any())
            {
                return string.Empty;
            }
            
            var sb = new StringBuilder();
            
            // Collect import namespaces for messages and enums only
            var importNamespaces = CollectImportNamespacesForServer(combineModel);
            
            // Add specific using statements for server generation
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            importNamespaces.Add("Microsoft.Extensions.Logging");
            
            GenerateUsingStatements(sb, importNamespaces);
            
            // Group services by CsharpNamespace
            var servicesByNamespace = model.Services
                .GroupBy(svc => svc.CsharpNamespace)
                .ToList();

            // Generate namespace blocks for each group
            foreach (var namespaceGroup in servicesByNamespace)
            {
                var namespaceValue = namespaceGroup.Key;
                sb.AppendLine($"namespace {namespaceValue}");
                sb.AppendLine("{");

                // Generate all services for this namespace
                foreach (var svc in namespaceGroup)
                {
                    GenerateServiceInterface(sb, svc, combineModel);
                    GenerateServiceImplementation(sb, svc, model, combineModel);
                }

                sb.AppendLine("}");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generate service interface for a given service
        /// </summary>
        private void GenerateServiceInterface(StringBuilder sb, ProtoService svc, ProtoModel messagesModel)
        {
            var serviceInterface = $"I{svc.Name}GrpcService";
            sb.AppendLine($"    public interface {serviceInterface}");
            sb.AppendLine("    {");
            
            foreach (var rpc in svc.Rpcs)
            {
                var rpcRequestType = messagesModel.FindCsharpTypeName(rpc.RequestType);
                var rpcResponseType= messagesModel.FindCsharpTypeName(rpc.ResponseType);
                sb.AppendLine($"        Task<{rpcResponseType}> {rpc.Name}({rpcRequestType} request);");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate service implementation class for a given service
        /// </summary>
        private void GenerateServiceImplementation(StringBuilder sb, ProtoService svc, ProtoModel model,
            ProtoModel combineModel)
        {
            var originalNamespace = svc.CsharpNamespace;
            var serviceInterface = $"I{svc.Name}GrpcService";
            var serviceClass = $"{svc.Name}NativeGrpcService";
            var baseClass = $"{originalNamespace}.{svc.Name}.{svc.Name}Base";
            
            sb.AppendLine($"    public class {serviceClass} : {baseClass}");
            sb.AppendLine("    {");
            sb.AppendLine($"        private readonly {serviceInterface} _instance;");
            sb.AppendLine();
            sb.AppendLine($"        public {serviceClass}({serviceInterface} instance)");
            sb.AppendLine("        {");
            sb.AppendLine("            _instance = instance;");
            sb.AppendLine("        }");
            sb.AppendLine();

            foreach (var rpc in svc.Rpcs)
            {
                GenerateServiceMethod(sb, rpc, model, originalNamespace, combineModel);
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate a single service method implementation
        /// </summary>
        private void GenerateServiceMethod(StringBuilder sb, ProtoRpc rpc, ProtoModel model, string originalNamespace,
            ProtoModel combineModel)
        {
            var rpcRequestFullType = combineModel.FindRpcFullTypename(rpc.RequestType);
            var rpcResponseFullType = combineModel.FindRpcFullTypename(rpc.ResponseType);
            
            var requestType = combineModel.FindCsharpTypeName(rpc.RequestType);
            var responseType = combineModel.FindCsharpTypeName(rpc.ResponseType);
            
            sb.AppendLine($"        public override async Task<{rpcResponseFullType}> {rpc.Name}({rpcRequestFullType} request, ServerCallContext context)");
            sb.AppendLine("        {");
            sb.AppendLine($"            var dtoRequest = new {requestType}();");
            
            // Map request fields
            var requestMessage = model.FindMessage(rpc.RequestType);
            if (requestMessage != null)
            {
                foreach (var field in requestMessage.Fields)
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    sb.AppendLine($"            dtoRequest.{propName} = request.{propName};");
                }
            }
            else
            {
                // Handle external types
            }
            
            sb.AppendLine($"            var dtoResponse = await _instance.{rpc.Name}(dtoRequest);");
            sb.AppendLine($"            var grpcResponse = new {rpcResponseFullType}();");
            
            // Map response fields
            var responseMessage = model.FindMessage(rpc.ResponseType);
            if (responseMessage != null)
            {
                foreach (var field in responseMessage.Fields)
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    sb.AppendLine($"            grpcResponse.{propName} = dtoResponse.{propName};");
                }
            }
            else
            {
                // Handle external types
            }
            
            sb.AppendLine("            return grpcResponse;");
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        private static string MapProtoCTypeToCSharp(string protoType, ProtoModel combineModel)
        {
            var csType= combineModel.FindCsharpTypeFullName(protoType);
            if (csType != null)
            {
                return csType;
            }
            
            return protoType switch
            {
                "int32" => "int",
                "int64" => "long",
                "uint32" => "uint",
                "uint64" => "ulong",
                "float" => "float",
                "double" => "double",
                "bool" => "bool",
                "string" => "string",
                "bytes" => "byte[]",
                "sfixed32" => "int",
                "Timestamp" => "DateTime",
                _ => protoType
            };
        }

        /// <summary>
        /// Collect namespaces from combined model messages and enums (for server generation)
        /// </summary>
        private HashSet<string> CollectImportNamespacesForServer(ProtoModel combinedModel)
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
        private void GenerateUsingStatements(StringBuilder sb, HashSet<string> namespaces)
        {
            // Add default using statements
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            
            // Add collected import namespaces (sorted for consistency)
            foreach (var ns in namespaces.OrderBy(x => x))
            {
                sb.AppendLine($"using {ns};");
            }
            
            sb.AppendLine();
        }
    }

    public class ProtoFileInfo
    {
        public string Path { get; set; }
        public string Content { get; set; }

        public string GetProtoFileName()
        {
            return System.IO.Path.GetFileNameWithoutExtension(Path);
        }
    }

    internal class ProtoImportResolver
    {
        private readonly Dictionary<string, ProtoModel> _parsedModels;
        private readonly Dictionary<string, ProtoFileInfo> _protoFiles;

        public ProtoImportResolver(IEnumerable<ProtoFileInfo> protoFiles)
        {
            _protoFiles = new Dictionary<string, ProtoFileInfo>();
            _parsedModels = new Dictionary<string, ProtoModel>();

            // Build a mapping of relative paths to proto files
            foreach (var protoFile in protoFiles)
            {
                var relativePath = NormalizeProtoPath(protoFile.Path);
                _protoFiles[relativePath] = protoFile;
            }
        }

        public ProtoModel EnrichModelWithImports(ProtoModel mainModel, string mainProtoPath)
        {
            var enrichedModel = new ProtoModel();

            // Copy original items
            foreach (var import in mainModel.Imports)
                enrichedModel.Imports.Add(import);
            
            foreach (var message in mainModel.Messages)
                enrichedModel.Messages.Add(message);
            
            foreach (var service in mainModel.Services)
                enrichedModel.Services.Add(service);
            
            foreach (var enumDef in mainModel.Enums)
                enrichedModel.Enums.Add(enumDef);

            // Add imported messages, services, and enums to the main model
            foreach (var importPath in mainModel.Imports)
            {
                var resolvedImport = ResolveImportPath(importPath, mainProtoPath);
                if (resolvedImport != null)
                {
                    var importedModel = GetOrParseModel(resolvedImport);
                    if (importedModel != null)
                    {
                        // Add imported messages with namespace prefix if needed
                        foreach (var message in importedModel.Messages)
                        {
                            if (!enrichedModel.Messages.Any(m => m.Name == message.Name))
                            {
                                enrichedModel.Messages.Add(message);
                            }
                        }

                        // Add imported enums
                        foreach (var enumDef in importedModel.Enums)
                        {
                            if (!enrichedModel.Enums.Any(e => e.Name == enumDef.Name))
                            {
                                enrichedModel.Enums.Add(enumDef);
                            }
                        }

                        // Note: Services are typically not imported, but we could add them if needed
                    }
                }
            }

            return enrichedModel;
        }

        public ProtoFileInfo ResolveImportPath(string importPath, string currentProtoPath)
        {
            // Try exact match first
            if (_protoFiles.TryGetValue(importPath, out var exactMatch))
            {
                return exactMatch;
            }

            // Try to resolve relative to current file's directory
            var currentDir = System.IO.Path.GetDirectoryName(NormalizeProtoPath(currentProtoPath));
            if (!string.IsNullOrEmpty(currentDir))
            {
                var relativePath = System.IO.Path.Combine(currentDir, importPath).Replace('\\', '/');
                if (_protoFiles.TryGetValue(relativePath, out var relativeMatch))
                {
                    return relativeMatch;
                }
            }

            // Try filename only match as fallback
            var importFileName = System.IO.Path.GetFileName(importPath);
            return _protoFiles.Values.FirstOrDefault(f => 
                System.IO.Path.GetFileName(f.Path).Equals(importFileName, StringComparison.OrdinalIgnoreCase));
        }

        public ProtoModel GetOrParseModel(ProtoFileInfo protoFile)
        {
            if (!_parsedModels.TryGetValue(protoFile.Path, out var model))
            {
                model = ProtoParser.ParseProtoText(protoFile.Content);
                _parsedModels[protoFile.Path] = model;
            }
            return model;
        }

        private string NormalizeProtoPath(string path)
        {
            // Extract relative path from full path
            // This handles cases where we have full absolute paths
            var segments = path.Replace('\\', '/').Split('/');
            
            // Find the "Protos" directory and take everything after it
            var protosIndex = Array.FindIndex(segments, s => s.Equals("Protos", StringComparison.OrdinalIgnoreCase));
            if (protosIndex >= 0 && protosIndex < segments.Length - 1)
            {
                return string.Join("/", segments.Skip(protosIndex + 1));
            }
            
            // Fallback: just take the filename
            return System.IO.Path.GetFileName(path);
        }
    }
}
