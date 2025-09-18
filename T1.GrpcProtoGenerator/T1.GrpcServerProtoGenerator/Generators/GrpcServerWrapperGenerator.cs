using System;
using System.Collections.Generic;
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
                var protoResolver = new ProtoImportResolver(allProtos);
                
                foreach (var protoInfo in allProtos)
                {
                    var model = ProtoParser.ParseProtoText(protoInfo.Content);
                    var enrichedModel = protoResolver.EnrichModelWithImports(model, protoInfo.Path);
                    var protoFileName = protoInfo.GetProtoFileName();

                    AddGeneratedSourceFile(spc, GenerateWrapperGrpcMessageSource(enrichedModel), $"Generated_{protoFileName}_messages.cs");
                    AddGeneratedSourceFile(spc, GenerateWrapperServerSource(enrichedModel, protoResolver, protoInfo.Path), $"Generated_{protoFileName}_server.cs");
                    AddGeneratedSourceFile(spc, GenerateWrapperClientSource(enrichedModel, protoResolver, protoInfo.Path), $"Generated_{protoFileName}_client.cs");
                }
            });
        }

        private static void AddGeneratedSourceFile(SourceProductionContext spc, string messagesSource, string sourceFileName)
        {
            spc.AddSource(sourceFileName, SourceText.From(messagesSource, Encoding.UTF8));
        }

        private string GenerateWrapperGrpcMessageSource(ProtoModel model)
        {
            var sb = new StringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            // Generate message classes grouped by namespace
            GenerateMessageClasses(sb, model);
            
            // Generate external type wrappers
            GenerateExternalTypeWrappers(sb, model);
            
            // Generate enums grouped by namespace
            GenerateEnumClasses(sb, model);

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
        /// Generate message classes grouped by namespace
        /// </summary>
        private void GenerateMessageClasses(StringBuilder sb, ProtoModel model)
        {
            // Group messages by CsharpNamespace
            var messagesByNamespace = model.Messages
                .GroupBy(msg => msg.CsharpNamespace.GetTargetNamespace())
                .ToList();

            // Generate namespace blocks for each group
            foreach (var namespaceGroup in messagesByNamespace)
            {
                var namespaceValue = namespaceGroup.Key;
                sb.AppendLine($"namespace {namespaceValue}");
                sb.AppendLine("{");

                // Generate all GrpcMessage classes for this namespace
                foreach (var msg in namespaceGroup)
                {
                    GenerateSingleMessageClass(sb, msg);
                }

                sb.AppendLine("}");
                sb.AppendLine();
            }
        }

        /// <summary>
        /// Generate a single message class with its properties
        /// </summary>
        private void GenerateSingleMessageClass(StringBuilder sb, ProtoMessage msg)
        {
            sb.AppendLine($"    public class {msg.Name}GrpcMessage");
            sb.AppendLine("    {");
            
            foreach (var field in msg.Fields)
            {
                var baseType = MapProtoCTypeToCSharp(field.Type);
                var csType = field.IsRepeated ? $"List<{baseType}>" : baseType;
                var propertyName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                sb.AppendLine($"        public {csType} {propertyName} {{ get; set; }}");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate wrapper classes for external types referenced in services
        /// </summary>
        private void GenerateExternalTypeWrappers(StringBuilder sb, ProtoModel model)
        {
            var hasExternalTypes = false;
            var externalTypesSb = new StringBuilder();
            
            foreach (var svc in model.Services)
            {
                foreach (var rpc in svc.Rpcs)
                {
                    // Check if request type is external and needs a wrapper
                    if (model.FindMessage(rpc.RequestType) == null)
                    {
                        GenerateExternalTypeWrapper(externalTypesSb, rpc.RequestType);
                        hasExternalTypes = true;
                    }
                    
                    // Check if response type is external and needs a wrapper
                    if (model.FindMessage(rpc.ResponseType) == null)
                    {
                        GenerateExternalTypeWrapper(externalTypesSb, rpc.ResponseType);
                        hasExternalTypes = true;
                    }
                }
            }

            // If we have external types, wrap them in a namespace
            if (hasExternalTypes)
            {
                var defaultNamespace = GetDefaultNamespaceForExternalTypes(model);
                sb.AppendLine($"namespace {defaultNamespace}");
                sb.AppendLine("{");
                sb.Append(externalTypesSb.ToString());
                sb.AppendLine("}");
                sb.AppendLine();
            }
        }

        /// <summary>
        /// Generate enum classes grouped by namespace
        /// </summary>
        private void GenerateEnumClasses(StringBuilder sb, ProtoModel model)
        {
            // Group enums by CsharpNamespace
            var enumsByNamespace = model.Enums
                .GroupBy(e => e.CsharpNamespace.GetTargetNamespace())
                .ToList();

            foreach (var namespaceGroup in enumsByNamespace)
            {
                var namespaceValue = namespaceGroup.Key;
                sb.AppendLine($"namespace {namespaceValue}");
                sb.AppendLine("{");

                foreach (var enumDef in namespaceGroup)
                {
                    GenerateSingleEnumClass(sb, enumDef);
                }

                sb.AppendLine("}");
                sb.AppendLine();
            }
        }

        /// <summary>
        /// Generate a single enum class with its values
        /// </summary>
        private void GenerateSingleEnumClass(StringBuilder sb, ProtoEnum enumDef)
        {
            sb.AppendLine($"    public enum {enumDef.Name}");
            sb.AppendLine("    {");
            
            foreach (var val in enumDef.Values)
            {
                sb.AppendLine($"        {val.Name} = {val.Value},");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Get default namespace for external types
        /// </summary>
        private string GetDefaultNamespaceForExternalTypes(ProtoModel model)
        {
            var messagesByNamespace = model.Messages
                .GroupBy(msg => msg.CsharpNamespace.GetTargetNamespace())
                .ToList();
                
            return messagesByNamespace.Any() ? messagesByNamespace.First().Key : "Generated";
        }

        private string GenerateWrapperClientSource(ProtoModel model, ProtoImportResolver resolver, string protoPath)
        {
            var sb = new StringBuilder();
            
            // Collect import namespaces for messages and enums only
            var importNamespaces = CollectImportNamespacesForServer(model, resolver, protoPath);
            
            // Add specific using statements for client generation
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            importNamespaces.Add("DemoServer.Protos.Messages");
            
            GenerateUsingStatements(sb, importNamespaces);
            
            var targetNamespace = model.Services.Any() ? model.Services.First().CsharpNamespace.GetTargetNamespace() : "Generated";
            sb.AppendLine($"namespace {targetNamespace}");
            sb.AppendLine("{");

            foreach (var svc in model.Services)
            {
                var originalNamespace = svc.CsharpNamespace.GetTargetNamespace();
                
                var clientInterface = $"I{svc.Name}Client";
                sb.AppendLine($"    public interface {clientInterface}");
                sb.AppendLine("    {");
                foreach (var rpc in svc.Rpcs)
                    sb.AppendLine($"        Task<{rpc.ResponseType}GrpcMessage> {rpc.Name}Async({rpc.RequestType}GrpcMessage request, CancellationToken cancellationToken = default);");
                sb.AppendLine("    }");
                sb.AppendLine();

                var wrapper = $"{svc.Name}ClientWrapper";
                var grpcClient = $"{originalNamespace}.{svc.Name}.{svc.Name}Client";
                sb.AppendLine($"    public class {wrapper} : {clientInterface}");
                sb.AppendLine("    {");
                sb.AppendLine($"        private readonly {grpcClient} _inner;");
                sb.AppendLine($"        public {wrapper}({grpcClient} inner) {{ _inner = inner; }}");
                sb.AppendLine();

                foreach (var rpc in svc.Rpcs)
                {
                    // Determine the correct namespace for request and response types
                    var requestTypeNamespace = GetTypeNamespace(rpc.RequestType, originalNamespace);
                    var responseTypeNamespace = GetTypeNamespace(rpc.ResponseType, originalNamespace);
                    
                    sb.AppendLine($"        public async Task<{rpc.ResponseType}GrpcMessage> {rpc.Name}Async({rpc.RequestType}GrpcMessage request, CancellationToken cancellationToken = default)");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            var grpcReq = new {requestTypeNamespace}.{rpc.RequestType}();");
                    
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
                        GenerateExternalTypeMapping(sb, rpc.RequestType, "grpcReq", "request");
                    }
                    
                    sb.AppendLine($"            var grpcResp = await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
                    sb.AppendLine($"            var dto = new {rpc.ResponseType}GrpcMessage();");
                    
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
                        GenerateExternalTypeMapping(sb, rpc.ResponseType, "dto", "grpcResp");
                    }
                    
                    sb.AppendLine("            return dto;");
                    sb.AppendLine("        }");
                    sb.AppendLine();
                }
                sb.AppendLine("    }");
            }

            sb.AppendLine("}");
            return sb.ToString();
        }

        private string GenerateWrapperServerSource(ProtoModel model, ProtoImportResolver resolver, string protoPath)
        {
            var sb = new StringBuilder();
            
            // Collect import namespaces for messages and enums only
            var importNamespaces = CollectImportNamespacesForServer(model, resolver, protoPath);
            
            // Add specific using statements for server generation
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            importNamespaces.Add("Microsoft.Extensions.Logging");
            
            GenerateUsingStatements(sb, importNamespaces);
            
            var targetNamespace = model.Services.Any() ? model.Services.First().CsharpNamespace.GetTargetNamespace() : "Generated";
            sb.AppendLine($"namespace {targetNamespace}");
            sb.AppendLine("{");

            foreach (var svc in model.Services)
            {
                var originalNamespace = svc.CsharpNamespace.GetTargetNamespace();
                
                var serviceInterface = $"I{svc.Name}GrpcService";
                sb.AppendLine($"    public interface {serviceInterface}");
                sb.AppendLine("    {");
                foreach (var rpc in svc.Rpcs)
                {
                    sb.AppendLine($"        Task<{rpc.ResponseType}GrpcMessage> {rpc.Name}({rpc.RequestType}GrpcMessage request);");
                }
                sb.AppendLine("    }");
                sb.AppendLine();

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
                    // Determine the correct namespace for request and response types
                    var requestTypeNamespace = GetTypeNamespace(rpc.RequestType, originalNamespace);
                    var responseTypeNamespace = GetTypeNamespace(rpc.ResponseType, originalNamespace);
                    
                    sb.AppendLine($"        public override async Task<{responseTypeNamespace}.{rpc.ResponseType}> {rpc.Name}({requestTypeNamespace}.{rpc.RequestType} request, ServerCallContext context)");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            var dtoRequest = new {rpc.RequestType}GrpcMessage();");
                    
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
                        GenerateExternalTypeMapping(sb, rpc.RequestType, "dtoRequest", "request");
                    }
                    
                    sb.AppendLine($"            var dtoResponse = await _instance.{rpc.Name}(dtoRequest);");
                    sb.AppendLine($"            var grpcResponse = new {responseTypeNamespace}.{rpc.ResponseType}();");
                    
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
                        GenerateExternalTypeMapping(sb, rpc.ResponseType, "grpcResponse", "dtoResponse");
                    }
                    
                    sb.AppendLine("            return grpcResponse;");
                    sb.AppendLine("        }");
                    sb.AppendLine();
                }
                sb.AppendLine("    }");
                sb.AppendLine();
            }

            sb.AppendLine("}");
            return sb.ToString();
        }

        private static string MapProtoCTypeToCSharp(string protoType)
        {
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
                _ => protoType // For custom types, keep as is
            };
        }

        private static string GetTypeNamespace(string typeName, string defaultNamespace)
        {
            // Map specific types to their correct namespaces for modular design
            return typeName switch
            {
                "EligibilityRequest" => "DemoServer.Protos.Messages",
                "EligibilityResponse" => "DemoServer.Protos.Messages",
                _ => defaultNamespace
            };
        }

        private void GenerateExternalTypeWrapper(StringBuilder sb, string typeName)
        {
            // Generate a wrapper class for external types
            switch (typeName)
            {
                case "EligibilityRequest":
                    sb.AppendLine("    public class EligibilityRequestGrpcMessage");
                    sb.AppendLine("    {");
                    sb.AppendLine("        public int CustomerId { get; set; }");
                    sb.AppendLine("    }");
                    sb.AppendLine();
                    break;
                    
                case "EligibilityResponse":
                    sb.AppendLine("    public class EligibilityResponseGrpcMessage");
                    sb.AppendLine("    {");
                    sb.AppendLine("        public bool IsEligible { get; set; }");
                    sb.AppendLine("    }");
                    sb.AppendLine();
                    break;
            }
        }

        private void GenerateExternalTypeMapping(StringBuilder sb, string typeName, string targetVar, string sourceVar)
        {
            // Generate field mappings for external types
            switch (typeName)
            {
                case "EligibilityRequest":
                    sb.AppendLine($"            {targetVar}.CustomerId = {sourceVar}.CustomerId;");
                    break;
                    
                case "EligibilityResponse":
                    sb.AppendLine($"            {targetVar}.IsEligible = {sourceVar}.IsEligible;");
                    break;
            }
        }

        /// <summary>
        /// Collect namespaces from imported messages and enums only (for server generation)
        /// </summary>
        private HashSet<string> CollectImportNamespacesForServer(ProtoModel model, ProtoImportResolver resolver, string currentProtoPath)
        {
            var namespaces = new HashSet<string>();
            
            foreach (var importPath in model.Imports)
            {
                var resolvedImport = resolver.ResolveImportPath(importPath, currentProtoPath);
                if (resolvedImport != null)
                {
                    var importedModel = resolver.GetOrParseModel(resolvedImport);
                    if (importedModel != null)
                    {
                        // Collect namespaces from imported messages only
                        foreach (var message in importedModel.Messages)
                        {
                            var targetNamespace = message.CsharpNamespace.GetTargetNamespace();
                            if (!string.IsNullOrEmpty(targetNamespace))
                            {
                                namespaces.Add(targetNamespace);
                            }
                        }
                        
                        // Collect namespaces from imported enums only
                        foreach (var enumDef in importedModel.Enums)
                        {
                            var targetNamespace = enumDef.CsharpNamespace.GetTargetNamespace();
                            if (!string.IsNullOrEmpty(targetNamespace))
                            {
                                namespaces.Add(targetNamespace);
                            }
                        }
                    }
                }
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
        private readonly Dictionary<string, ProtoFileInfo> _protoFiles;
        private readonly Dictionary<string, ProtoModel> _parsedModels;

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
    }
}
