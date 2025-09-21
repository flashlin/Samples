using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using T1.GrpcProtoGenerator.Common;

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
                var logger = InitializeLogger(spc);
                logger.LogWarning($"Starting source generation for {allProtos.Length} proto files");
                
                var combinedModel = new ProtoModelResolver().CreateCombinedModel(allProtos);
                
                GenerateMessageFiles(spc, combinedModel, logger);
                GenerateEnumFiles(spc, combinedModel, logger);
                GenerateServiceFiles(spc, allProtos, combinedModel, logger);
                
                logger.LogInfo("Source generation completed successfully");
            });
        }

        /// <summary>
        /// Initialize logger for source generation
        /// </summary>
        private ISourceGeneratorLogger InitializeLogger(SourceProductionContext spc)
        {
            return new SourceGeneratorLogger(spc.ReportDiagnostic, nameof(GrpcServerWrapperGenerator));
        }

        /// <summary>
        /// Generate message files for all messages in the combined model
        /// </summary>
        private void GenerateMessageFiles(SourceProductionContext spc, ProtoModel combinedModel, ISourceGeneratorLogger logger)
        {
            logger.LogDebug($"Generating {combinedModel.Messages.Count} message files");
            
            foreach (var messageModel in combinedModel.Messages)
            {
                AddGeneratedSourceFile(spc, GenerateWrapperGrpcMessageSource(messageModel, combinedModel), 
                    $"Generated_message_{messageModel.Name}.cs");
            }
        }

        /// <summary>
        /// Generate enum files for all enums in the combined model
        /// </summary>
        private void GenerateEnumFiles(SourceProductionContext spc, ProtoModel combinedModel, ISourceGeneratorLogger logger)
        {
            logger.LogDebug($"Generating {combinedModel.Enums.Count} enum files");
            
            foreach (var enumModel in combinedModel.Enums)
            {
                AddGeneratedSourceFile(spc, GenerateWrapperGrpcEnumSource(enumModel), 
                    $"Generated_enum_{enumModel.Name}.cs");
            }
        }

        /// <summary>
        /// Generate server and client files for all proto files
        /// </summary>
        private void GenerateServiceFiles(SourceProductionContext spc, ImmutableArray<ProtoFileInfo> allProtos, 
            ProtoModel combinedModel, ISourceGeneratorLogger logger)
        {
            logger.LogDebug($"Generating service files for {allProtos.Length} proto files");
            
            AddGeneratedSourceFile(spc, GenerateGrpcUtilsSource(), $"Generated_grpcUtils.cs");
            
            foreach (var protoInfo in allProtos)
            {
                var model = ProtoParser.ParseProtoText(protoInfo.Content, protoInfo.Path);
                var protoFileName = protoInfo.GetProtoFileName();
                
                logger.LogDebug($"Generating server and client files for {protoFileName}");
                
                // Generate server and client files per proto file
                AddGeneratedSourceFile(spc, GenerateWrapperServerSource(model, combinedModel), 
                    $"Generated_{protoFileName}_server.cs");
                AddGeneratedSourceFile(spc, GenerateWrapperClientSource(model, combinedModel), 
                    $"Generated_{protoFileName}_client.cs");
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


        private string GenerateWrapperGrpcMessageSource(ProtoMessage messageModel, ProtoModel combinedModel)
        {
            var sb = new IndentStringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            sb.WriteLine($"namespace {messageModel.CsharpNamespace}");
            sb.WriteLine("{");
            sb.Indent++;

            GenerateSingleMessageClass(sb, messageModel, combinedModel);

            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();

            return sb.ToString();
        }

        private string GenerateWrapperGrpcEnumSource(ProtoEnum enumModel)
        {
            var sb = new IndentStringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            sb.WriteLine($"namespace {enumModel.CsharpNamespace}");
            sb.WriteLine("{");
            sb.Indent++;

            GenerateSingleEnumClass(sb, enumModel);

            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();

            return sb.ToString();
        }

        /// <summary>
        /// Generate basic using statements for message source files
        /// </summary>
        private void GenerateBasicUsingStatements(IndentStringBuilder sb)
        {
            sb.WriteLine("#nullable enable");
            sb.WriteLine("using System;");
            sb.WriteLine("using System.Collections.Generic;");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate a single message class with its properties
        /// </summary>
        private void GenerateSingleMessageClass(IndentStringBuilder sb, ProtoMessage msg, ProtoModel combineModel)
        {
            if (!ShouldGenerateMessageClass(msg))
            {
                return;
            }
            
            GenerateMessageClassDeclaration(sb, msg);
            GenerateMessageClassProperties(sb, msg, combineModel);
            GenerateMessageClassClosing(sb);
        }

        /// <summary>
        /// Check if message class should be generated
        /// </summary>
        private bool ShouldGenerateMessageClass(ProtoMessage msg)
        {
            // Skip generating DTO for Google Void/Null types as they correspond to no C# class needed
            return !msg.Name.Equals("Void", StringComparison.OrdinalIgnoreCase) && 
                   !msg.Name.Equals("Null", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Generate message class declaration and opening brace
        /// </summary>
        private void GenerateMessageClassDeclaration(IndentStringBuilder sb, ProtoMessage msg)
        {
            sb.WriteLine($"public class {msg.GetCsharpTypeName()}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all properties for the message class
        /// </summary>
        private void GenerateMessageClassProperties(IndentStringBuilder sb, ProtoMessage msg, ProtoModel combineModel)
        {
            foreach (var field in msg.Fields)
            {
                GenerateSingleMessageProperty(sb, field, combineModel);
            }
        }

        /// <summary>
        /// Generate a single property for the message class
        /// </summary>
        private void GenerateSingleMessageProperty(IndentStringBuilder sb, ProtoField field, ProtoModel combineModel)
        {
            var propertyInfo = CreateMessagePropertyInfo(field, combineModel);
            var propertyDeclaration = CreatePropertyDeclaration(propertyInfo);
            sb.WriteLine(propertyDeclaration);
        }

        /// <summary>
        /// Create property information for message field
        /// </summary>
        private MessagePropertyInfo CreateMessagePropertyInfo(ProtoField field, ProtoModel combineModel)
        {
            var baseType = MapProtoCTypeToCSharp(field.Type, combineModel);
            var finalType = ApplyFieldModifiers(baseType, field);
            var propertyName = FormatPropertyName(field.Name);

            return new MessagePropertyInfo
            {
                Type = finalType,
                Name = propertyName,
                IsOptional = field.IsOption
            };
        }

        /// <summary>
        /// Apply field modifiers to the base type
        /// </summary>
        private string ApplyFieldModifiers(string baseType, ProtoField field)
        {
            if (field.IsOption)
            {
                baseType += "?";
            }
            
            return field.IsRepeated ? $"List<{baseType}>" : baseType;
        }

        /// <summary>
        /// Format field name to property name (PascalCase)
        /// </summary>
        private string FormatPropertyName(string fieldName)
        {
            return char.ToUpper(fieldName[0]) + fieldName.Substring(1);
        }

        /// <summary>
        /// Create property declaration string
        /// </summary>
        private string CreatePropertyDeclaration(MessagePropertyInfo propertyInfo)
        {
            var modifier = propertyInfo.IsOptional ? "" : "required ";
            return $"public {modifier}{propertyInfo.Type} {propertyInfo.Name} {{ get; set; }}";
        }

        /// <summary>
        /// Generate message class closing brace
        /// </summary>
        private void GenerateMessageClassClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate a single enum class with its values
        /// </summary>
        private void GenerateSingleEnumClass(IndentStringBuilder sb, ProtoEnum enumDef)
        {
            sb.WriteLine($"public enum {enumDef.GetCsharpTypeName()}");
            sb.WriteLine("{");
            sb.Indent++;
            
            foreach (var val in enumDef.Values)
            {
                sb.WriteLine($"{val.Name} = {val.Value},");
            }
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        private string GenerateWrapperClientSource(ProtoModel model, ProtoModel combinedModel)
        {
            if (!ValidateClientSourceGeneration(model))
            {
                return string.Empty;
            }
            
            var sb = new IndentStringBuilder();
            
            // Setup using statements
            SetupClientSourceUsingStatements(sb, combinedModel);
            
            // Group and generate services by namespace
            var servicesByNamespace = GroupServicesByNamespace(model);
            GenerateClientNamespaceBlocks(sb, servicesByNamespace, combinedModel);

            return sb.ToString();
        }

        /// <summary>
        /// Validate if client source generation is needed
        /// </summary>
        private bool ValidateClientSourceGeneration(ProtoModel model)
        {
            return model.Services.Any();
        }

        /// <summary>
        /// Setup using statements for client source generation
        /// </summary>
        private void SetupClientSourceUsingStatements(IndentStringBuilder sb, ProtoModel combinedModel)
        {
            var importNamespaces = CollectImportNamespacesForServer(combinedModel);
            AddClientSpecificNamespaces(importNamespaces);
            GenerateUsingStatements(sb, importNamespaces);
        }

        /// <summary>
        /// Add client-specific namespaces to the import list
        /// </summary>
        private void AddClientSpecificNamespaces(HashSet<string> importNamespaces)
        {
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            importNamespaces.Add("Microsoft.Extensions.DependencyInjection");
            importNamespaces.Add("Microsoft.Extensions.Options");
            importNamespaces.Add("Grpc.Net.Client");
            importNamespaces.Add("T1.GrpcProtoGenerator");
        }

        /// <summary>
        /// Generate client namespace blocks with their contained services
        /// </summary>
        private void GenerateClientNamespaceBlocks(IndentStringBuilder sb, 
            List<IGrouping<string, ProtoService>> servicesByNamespace, 
            ProtoModel combinedModel)
        {
            foreach (var namespaceGroup in servicesByNamespace)
            {
                GenerateClientNamespaceDeclaration(sb, namespaceGroup.Key);
                GenerateClientServicesInNamespace(sb, namespaceGroup, combinedModel);
                GenerateClientNamespaceClosing(sb);
            }
        }

        /// <summary>
        /// Generate client namespace declaration
        /// </summary>
        private void GenerateClientNamespaceDeclaration(IndentStringBuilder sb, string namespaceValue)
        {
            sb.WriteLine($"namespace {namespaceValue}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all client services within a namespace
        /// </summary>
        private void GenerateClientServicesInNamespace(IndentStringBuilder sb, 
            IGrouping<string, ProtoService> namespaceGroup, 
            ProtoModel combinedModel)
        {
            foreach (var svc in namespaceGroup)
            {
                GenerateClientInterface(sb, svc);
                GenerateClientWrapper(sb, svc, combinedModel);
                GenerateClientConfig(sb, svc);
                GenerateClientExtension(sb, svc);
            }
        }

        /// <summary>
        /// Generate client namespace closing brace
        /// </summary>
        private void GenerateClientNamespaceClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate client interface for a given service
        /// </summary>
        private void GenerateClientInterface(IndentStringBuilder sb, ProtoService svc)
        {
            GenerateClientInterfaceDeclaration(sb, svc);
            GenerateClientInterfaceMethods(sb, svc);
            GenerateClientInterfaceClosing(sb);
        }

        /// <summary>
        /// Generate client interface declaration and opening brace
        /// </summary>
        private void GenerateClientInterfaceDeclaration(IndentStringBuilder sb, ProtoService svc)
        {
            var clientInterface = $"I{svc.Name}GrpcClient";
            sb.WriteLine($"public interface {clientInterface}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all client interface methods for the service
        /// </summary>
        private void GenerateClientInterfaceMethods(IndentStringBuilder sb, ProtoService svc)
        {
            foreach (var rpc in svc.Rpcs)
            {
                GenerateSingleClientInterfaceMethod(sb, rpc);
            }
        }

        /// <summary>
        /// Generate a single client interface method signature
        /// </summary>
        private void GenerateSingleClientInterfaceMethod(IndentStringBuilder sb, ProtoRpc rpc)
        {
            var methodSignature = CreateClientInterfaceMethodSignature(rpc);
            sb.WriteLine($"{methodSignature.ReturnType} {rpc.Name}Async({methodSignature.Parameters}CancellationToken cancellationToken = default);");
        }

        /// <summary>
        /// Create method signature information for client interface method
        /// </summary>
        private ClientInterfaceMethodSignature CreateClientInterfaceMethodSignature(ProtoRpc rpc)
        {
            return new ClientInterfaceMethodSignature
            {
                Parameters = GenerateClientInterfaceMethodParameters(rpc),
                ReturnType = GenerateClientInterfaceMethodReturnType(rpc)
            };
        }

        /// <summary>
        /// Generate parameters for client interface method
        /// </summary>
        private string GenerateClientInterfaceMethodParameters(ProtoRpc rpc)
        {
            // Skip parameter if request type is "Null" or "google.protobuf.Empty"
            if (rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase) ||
                rpc.RequestType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase))
            {
                return "";
            }
            
            return $"{rpc.RequestType}GrpcDto request, ";
        }

        /// <summary>
        /// Generate return type for client interface method
        /// </summary>
        private string GenerateClientInterfaceMethodReturnType(ProtoRpc rpc)
        {
            // Use Task instead of Task<T> if response type is "Void" or "google.protobuf.Empty"
            if (rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase) ||
                rpc.ResponseType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase))
            {
                return "Task";
            }
            
            return $"Task<{rpc.ResponseType}GrpcDto>";
        }

        /// <summary>
        /// Generate client interface closing brace
        /// </summary>
        private void GenerateClientInterfaceClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate client wrapper class for a given service
        /// </summary>
        private void GenerateClientWrapper(IndentStringBuilder sb, ProtoService svc, ProtoModel combineModel)
        {
            var originalNamespace = svc.CsharpNamespace;
            var clientInterface = $"I{svc.Name}GrpcClient";
            var wrapper = $"{svc.Name}GrpcClient";
            var grpcClient = $"{originalNamespace}.{svc.Name}.{svc.Name}Client";
            
            sb.WriteLine($"public class {wrapper} : {clientInterface}");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"private readonly {grpcClient} _inner;");
            sb.WriteLine($"public {wrapper}({grpcClient} inner) {{ _inner = inner; }}");
            sb.WriteLine();

            foreach (var rpc in svc.Rpcs)
            {
                GenerateClientMethod(sb, rpc, combineModel);
            }
            
            // Generate conversion methods for custom types
            GenerateClientConversionMethods(sb, svc, combineModel);
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate a single client method implementation
        /// </summary>
        private void GenerateClientMethod(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var clientMethodInfo = GetClientMethodInfo(rpc);
            
            // Generate method signature
            GenerateClientMethodSignature(sb, rpc, clientMethodInfo);
            
            // Generate method body
            sb.WriteLine("{");
            sb.Indent++;
            
            // Generate request mapping
            GenerateClientRequestMapping(sb, rpc, clientMethodInfo, combineModel);
            
            // Generate service call and response handling
            GenerateClientServiceCallAndResponse(sb, rpc, clientMethodInfo, combineModel);
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Get client method information including type flags
        /// </summary>
        private ClientMethodInfo GetClientMethodInfo(ProtoRpc rpc)
        {
            return new ClientMethodInfo
            {
                IsNullRequest = rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase) ||
                               rpc.RequestType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase),
                IsVoidResponse = rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase) ||
                                rpc.ResponseType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase)
            };
        }

        /// <summary>
        /// Generate client method signature
        /// </summary>
        private void GenerateClientMethodSignature(IndentStringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo)
        {
            var parameterPart = methodInfo.IsNullRequest ? "" : $"{rpc.RequestType}GrpcDto request, ";
            var returnType = methodInfo.IsVoidResponse ? "Task" : $"Task<{rpc.ResponseType}GrpcDto>";
            
            sb.WriteLine($"public async {returnType} {rpc.Name}Async({parameterPart}CancellationToken cancellationToken = default)");
        }

        /// <summary>
        /// Generate client request mapping from DTO to gRPC
        /// </summary>
        private void GenerateClientRequestMapping(IndentStringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (!methodInfo.IsNullRequest)
            {
                GenerateClientRequestObjectMapping(sb, rpc, combineModel);
            }
            else
            {
                GenerateClientNullRequestMapping(sb, rpc, combineModel);
            }
        }

        /// <summary>
        /// Generate client request object mapping for non-null requests
        /// </summary>
        private void GenerateClientRequestObjectMapping(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var requestMessage = combineModel.FindMessage(rpc.RequestType);
            var requestFullType = combineModel.FindRpcFullTypename(rpc.RequestType);
            sb.WriteLine($"var grpcReq = new {requestFullType}");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Only generate field mappings if the message exists and has fields
            if (requestMessage != null && requestMessage.Fields.Any())
            {
                GenerateDtoFieldToRpcMappings(sb, requestMessage.Fields, "request");
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            
            // Handle repeated fields separately after object initialization
            if (requestMessage != null && requestMessage.Fields.Any(f => f.IsRepeated))
            {
                foreach (var field in requestMessage.Fields.Where(f => f.IsRepeated))
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    GenerateRepeatedFieldMapping(sb, field, propName, "request", "grpcReq");
                }
            }
        }

        /// <summary>
        /// Generate client request mapping for null requests
        /// </summary>
        private void GenerateClientNullRequestMapping(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var requestFullType = combineModel.FindRpcFullTypename(rpc.RequestType);
            sb.WriteLine($"var grpcReq = new {requestFullType}();");
        }

        /// <summary>
        /// Generate client service call and response handling
        /// </summary>
        private void GenerateClientServiceCallAndResponse(IndentStringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (methodInfo.IsVoidResponse)
            {
                GenerateClientVoidServiceCall(sb, rpc);
            }
            else
            {
                GenerateClientNormalServiceCall(sb, rpc, combineModel);
            }
        }

        /// <summary>
        /// Generate client service call for void response methods
        /// </summary>
        private void GenerateClientVoidServiceCall(IndentStringBuilder sb, ProtoRpc rpc)
        {
            sb.WriteLine($"await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
        }

        /// <summary>
        /// Generate client service call for normal response methods
        /// </summary>
        private void GenerateClientNormalServiceCall(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            sb.WriteLine($"var grpcResp = await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
            
            // Generate response mapping
            GenerateClientResponseMapping(sb, rpc, combineModel);
        }

        /// <summary>
        /// Generate client response mapping from gRPC to DTO
        /// </summary>
        private void GenerateClientResponseMapping(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var responseMessage = combineModel.FindMessage(rpc.ResponseType);
            sb.WriteLine($"var dto = new {rpc.ResponseType}GrpcDto");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Only generate field mappings if the message exists and has fields
            if (responseMessage != null && responseMessage.Fields.Any())
            {
                GenerateRpcFieldToDtoMappings(sb, responseMessage.Fields, "grpcResp");
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            
            // Handle repeated fields separately after object initialization
            if (responseMessage != null && responseMessage.Fields.Any(f => f.IsRepeated))
            {
                foreach (var field in responseMessage.Fields.Where(f => f.IsRepeated))
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    GenerateClientRepeatedFieldMapping(sb, field, propName, "grpcResp", "dto");
                }
            }
            
            sb.WriteLine("return dto;");
        }

        /// <summary>
        /// Generate gRPC client configuration class
        /// </summary>
        private void GenerateClientConfig(IndentStringBuilder sb, ProtoService svc)
        {
            var configClassName = $"{svc.Name}GrpcConfig";
            
            sb.WriteLine($"public class {configClassName}");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine("public string ServerUrl { get; set; } = \"https://localhost:7001\";");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate gRPC client extension class for dependency injection
        /// </summary>
        private void GenerateClientExtension(IndentStringBuilder sb, ProtoService svc)
        {
            var extensionClassName = $"{svc.Name}GrpcExtension";
            var configClassName = $"{svc.Name}GrpcConfig";
            var clientInterface = $"I{svc.Name}GrpcClient";
            var wrapper = $"{svc.Name}GrpcClient";
            var originalNamespace = svc.CsharpNamespace;
            var grpcClient = $"{originalNamespace}.{svc.Name}.{svc.Name}Client";
            
            sb.WriteLine($"public static class {extensionClassName}");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Generate AddXxxGrpcSdk extension method
            sb.WriteLine($"public static void Add{svc.Name}GrpcSdk(this IServiceCollection services)");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Register the original gRPC Client with dedicated channel
            sb.WriteLine($"services.AddTransient<{grpcClient}>(provider =>");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"var config = provider.GetRequiredService<IOptions<{configClassName}>>();");
            sb.WriteLine("var channel = GrpcChannel.ForAddress(config.Value.ServerUrl);");
            sb.WriteLine($"return new {grpcClient}(channel);");
            sb.Indent--;
            sb.WriteLine("});");
            
            // Register the wrapper gRPC Client interface and implementation
            sb.WriteLine($"services.AddTransient<{clientInterface}, {wrapper}>();");
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        private string GenerateGrpcUtilsSource()
        {
            var sb = new IndentStringBuilder();
            
            sb.WriteLine("using System;");
            sb.WriteLine("using Google.Protobuf.WellKnownTypes;");
            
            sb.WriteLine("namespace T1.GrpcProtoGenerator");
            sb.WriteLine("{");
            sb.Indent++;
            GenerateGrpcHelperClass(sb);
            sb.Indent--;
            sb.WriteLine("}");
            return sb.ToString();
        }

        private string GenerateWrapperServerSource(ProtoModel model, ProtoModel combineModel)
        {
            if (!ValidateServerSourceGeneration(model))
            {
                return string.Empty;
            }
            
            var sb = new IndentStringBuilder();
            
            // Setup using statements
            SetupServerSourceUsingStatements(sb, combineModel);
            
            // Group and generate services by namespace
            var servicesByNamespace = GroupServicesByNamespace(model);
            GenerateNamespaceBlocks(sb, servicesByNamespace, combineModel);

            return sb.ToString();
        }

        /// <summary>
        /// Validate if server source generation is needed
        /// </summary>
        private bool ValidateServerSourceGeneration(ProtoModel model)
        {
            return model.Services.Any();
        }

        /// <summary>
        /// Setup using statements for server source generation
        /// </summary>
        private void SetupServerSourceUsingStatements(IndentStringBuilder sb, ProtoModel combineModel)
        {
            var importNamespaces = CollectImportNamespacesForServer(combineModel);
            AddServerSpecificNamespaces(importNamespaces);
            GenerateUsingStatements(sb, importNamespaces);
        }

        /// <summary>
        /// Add server-specific namespaces to the import list
        /// </summary>
        private void AddServerSpecificNamespaces(HashSet<string> importNamespaces)
        {
            importNamespaces.Add("System.Threading");
            importNamespaces.Add("System.Threading.Tasks");
            importNamespaces.Add("Grpc.Core");
            importNamespaces.Add("Microsoft.Extensions.Logging");
            importNamespaces.Add("T1.GrpcProtoGenerator");
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
        /// Generate namespace blocks with their contained services
        /// </summary>
        private void GenerateNamespaceBlocks(IndentStringBuilder sb, 
            List<IGrouping<string, ProtoService>> servicesByNamespace, 
            ProtoModel combineModel)
        {
            foreach (var namespaceGroup in servicesByNamespace)
            {
                GenerateNamespaceDeclaration(sb, namespaceGroup.Key);
                GenerateServicesInNamespace(sb, namespaceGroup, combineModel);
                GenerateNamespaceClosing(sb);
            }
        }

        /// <summary>
        /// Generate namespace declaration
        /// </summary>
        private void GenerateNamespaceDeclaration(IndentStringBuilder sb, string namespaceValue)
        {
            sb.WriteLine($"namespace {namespaceValue}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all services within a namespace
        /// </summary>
        private void GenerateServicesInNamespace(IndentStringBuilder sb, 
            IGrouping<string, ProtoService> namespaceGroup, 
            ProtoModel combineModel)
        {
            foreach (var svc in namespaceGroup)
            {
                GenerateServiceInterface(sb, svc, combineModel);
                GenerateServiceImplementation(sb, svc, combineModel);
            }
        }

        /// <summary>
        /// Generate GrpcHelper static class with extension methods
        /// </summary>
        private void GenerateGrpcHelperClass(IndentStringBuilder sb)
        {
            sb.WriteLine("public static class GrpcHelper");
            sb.WriteLine("{");
            sb.Indent++;
            
            sb.WriteLine("public static Timestamp ToGrpcTimestamp(this DateTime dt)");
            sb.WriteLine("{");
            sb.Indent++;
            
            sb.WriteLine("var utcTime = dt;");
            sb.WriteLine("if (dt.Kind == DateTimeKind.Local)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine("utcTime = DateTime.SpecifyKind(dt, DateTimeKind.Utc);");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine("return Timestamp.FromDateTime(utcTime);");
            sb.Indent--;
            sb.WriteLine("}");
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate namespace closing brace
        /// </summary>
        private void GenerateNamespaceClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate service interface for a given service
        /// </summary>
        private void GenerateServiceInterface(IndentStringBuilder sb, ProtoService svc, ProtoModel messagesModel)
        {
            GenerateInterfaceDeclaration(sb, svc);
            GenerateInterfaceMethods(sb, svc, messagesModel);
            GenerateInterfaceClosing(sb);
        }

        /// <summary>
        /// Generate interface declaration and opening brace
        /// </summary>
        private void GenerateInterfaceDeclaration(IndentStringBuilder sb, ProtoService svc)
        {
            var serviceInterface = $"I{svc.Name}GrpcService";
            sb.WriteLine($"public interface {serviceInterface}");
            sb.WriteLine("{");
            sb.Indent++;
        }

        /// <summary>
        /// Generate all interface methods for the service
        /// </summary>
        private void GenerateInterfaceMethods(IndentStringBuilder sb, ProtoService svc, ProtoModel messagesModel)
        {
            foreach (var rpc in svc.Rpcs)
            {
                GenerateSingleInterfaceMethod(sb, rpc, messagesModel);
            }
        }

        /// <summary>
        /// Generate a single interface method signature
        /// </summary>
        private void GenerateSingleInterfaceMethod(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel messagesModel)
        {
            var methodSignature = CreateInterfaceMethodSignature(rpc, messagesModel);
            sb.WriteLine($"{methodSignature.ReturnType} {rpc.Name}({methodSignature.Parameters});");
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
            if (rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase) ||
                rpc.RequestType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase))
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
            if (rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase) ||
                rpc.ResponseType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase))
            {
                return "Task";
            }
            
            return $"Task<{rpcResponseType}>";
        }

        /// <summary>
        /// Generate interface closing brace
        /// </summary>
        private void GenerateInterfaceClosing(IndentStringBuilder sb)
        {
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate service implementation class for a given service
        /// </summary>
        private void GenerateServiceImplementation(IndentStringBuilder sb, ProtoService svc,
            ProtoModel combineModel)
        {
            var originalNamespace = svc.CsharpNamespace;
            var serviceInterface = $"I{svc.Name}GrpcService";
            var serviceClass = $"{svc.Name}NativeGrpcService";
            var baseClass = $"{originalNamespace}.{svc.Name}.{svc.Name}Base";
            
            sb.WriteLine($"public class {serviceClass} : {baseClass}");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"private readonly {serviceInterface} _instance;");
            sb.WriteLine();
            sb.WriteLine($"public {serviceClass}({serviceInterface} instance)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine("_instance = instance;");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();

            foreach (var rpc in svc.Rpcs)
            {
                GenerateServiceMethod(sb, rpc, combineModel);
            }
            
            // Generate conversion methods for custom types
            GenerateServerConversionMethods(sb, svc, combineModel);
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate a single service method implementation
        /// </summary>
        private void GenerateServiceMethod(IndentStringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var methodInfo = GetServiceMethodInfo(rpc, combineModel);
            
            // Generate method signature
            GenerateServiceMethodSignature(sb, rpc, methodInfo);
            
            // Generate method body
            sb.WriteLine("{");
            sb.Indent++;
            
            // Generate request mapping
            GenerateRequestMapping(sb, rpc, methodInfo, combineModel);
            
            // Generate service call and response handling
            GenerateServiceCallAndResponse(sb, rpc, methodInfo, combineModel);
            
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Get service method information including type names and flags
        /// </summary>
        private ServiceMethodInfo GetServiceMethodInfo(ProtoRpc rpc, ProtoModel combineModel)
        {
            return new ServiceMethodInfo
            {
                RequestFullType = combineModel.FindRpcFullTypename(rpc.RequestType),
                ResponseFullType = combineModel.FindRpcFullTypename(rpc.ResponseType),
                RequestType = combineModel.FindCsharpTypeName(rpc.RequestType),
                ResponseType = combineModel.FindCsharpTypeName(rpc.ResponseType),
                IsNullRequest = rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase) ||
                               rpc.RequestType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase),
                IsVoidResponse = rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase) ||
                                rpc.ResponseType.Equals("google.protobuf.Empty", StringComparison.OrdinalIgnoreCase)
            };
        }

        /// <summary>
        /// Generate service method signature
        /// </summary>
        private void GenerateServiceMethodSignature(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo)
        {
            sb.WriteLine($"public override async Task<{methodInfo.ResponseFullType}> {rpc.Name}({methodInfo.RequestFullType} request, ServerCallContext context)");
        }

        /// <summary>
        /// Generate request parameter mapping from gRPC to DTO
        /// </summary>
        private void GenerateRequestMapping(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (!methodInfo.IsNullRequest)
            {
                var requestMessage = combineModel.FindMessage(rpc.RequestType);
                sb.WriteLine($"var dtoRequest = new {methodInfo.RequestType}");
                sb.WriteLine("{");
                sb.Indent++;
                
                // Only generate field mappings if the message exists and has fields
                if (requestMessage != null && requestMessage.Fields.Any())
                {
                    GenerateRpcFieldToDtoMappings(sb, requestMessage.Fields, "request");
                }
                
                sb.Indent--;
                sb.WriteLine("};");
                
                // Handle repeated fields separately after object initialization (gRPC to DTO)
                if (requestMessage != null && requestMessage.Fields.Any(f => f.IsRepeated))
                {
                    foreach (var field in requestMessage.Fields.Where(f => f.IsRepeated))
                    {
                        var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                        GenerateServerRequestRepeatedFieldMapping(sb, field, propName, "request", "dtoRequest");
                    }
                }
            }
        }

        /// <summary>
        /// Generate service call and response handling
        /// </summary>
        private void GenerateServiceCallAndResponse(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (methodInfo.IsVoidResponse)
            {
                GenerateVoidServiceCall(sb, rpc, methodInfo);
            }
            else
            {
                GenerateNormalServiceCall(sb, rpc, methodInfo, combineModel);
            }
        }

        /// <summary>
        /// Generate service call for void response methods
        /// </summary>
        private void GenerateVoidServiceCall(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo)
        {
            // Call service method without expecting return value
            if (methodInfo.IsNullRequest)
            {
                sb.WriteLine($"await _instance.{rpc.Name}();");
            }
            else
            {
                sb.WriteLine($"await _instance.{rpc.Name}(dtoRequest);");
            }
            
            // Return empty response instance
            sb.WriteLine($"return new {methodInfo.ResponseFullType}();");
        }

        /// <summary>
        /// Generate service call for normal response methods
        /// </summary>
        private void GenerateNormalServiceCall(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            // Call service method and get return value
            if (methodInfo.IsNullRequest)
            {
                sb.WriteLine($"var dtoResponse = await _instance.{rpc.Name}();");
            }
            else
            {
                sb.WriteLine($"var dtoResponse = await _instance.{rpc.Name}(dtoRequest);");
            }
            
            // Generate response mapping
            GenerateResponseMapping(sb, rpc, methodInfo, combineModel);
        }

        /// <summary>
        /// Generate response mapping from DTO to gRPC
        /// </summary>
        private void GenerateResponseMapping(IndentStringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            var responseMessage = combineModel.FindMessage(rpc.ResponseType);
            sb.WriteLine($"var grpcResponse = new {methodInfo.ResponseFullType}");
            sb.WriteLine("{");
            sb.Indent++;
            
            // Only generate field mappings if the message exists and has fields
            if (responseMessage != null && responseMessage.Fields.Any())
            {
                GenerateDtoFieldToRpcMappings(sb, responseMessage.Fields, "dtoResponse");
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            
            // Handle repeated fields separately after object initialization
            if (responseMessage != null && responseMessage.Fields.Any(f => f.IsRepeated))
            {
                foreach (var field in responseMessage.Fields.Where(f => f.IsRepeated))
                {
                    var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                    GenerateRepeatedFieldMapping(sb, field, propName, "dtoResponse", "grpcResponse");
                }
            }
            
            sb.WriteLine("return grpcResponse;");
        }

        /// <summary>
        /// Generate field mappings for object initializer
        /// </summary>
        private void GenerateRpcFieldToDtoMappings(IndentStringBuilder sb, List<ProtoField> fields, string sourceObject)
        {
            for (int i = 0; i < fields.Count; i++)
            {
                var field = fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < fields.Count - 1 ? "," : "";
                
                // Handle repeated fields with empty list initialization
                if (field.IsRepeated)
                {
                    var elementType = IsPrimitiveType(field.Type) 
                        ? GetCsharpElementType(field.Type) 
                        : $"{field.Type}GrpcDto";
                    sb.WriteLine($"{propName} = new List<{elementType}>()" + comma);
                    continue;
                }
                
                // Check if field type is timestamp and needs conversion
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = {sourceObject}.{propName}.ToDateTime(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = {sourceObject}.{propName}{comma}");
                }
            }
        }
        
        
        /// <summary>
        /// Generate field mappings for object initializer
        /// </summary>
        private void GenerateDtoFieldToRpcMappings(IndentStringBuilder sb, List<ProtoField> fields, string sourceObject)
        {
            for (int i = 0; i < fields.Count; i++)
            {
                var field = fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < fields.Count - 1 ? "," : "";
                
                // Skip repeated fields in object initializer - they will be handled separately
                if (field.IsRepeated)
                {
                    continue;
                }
                
                // Check if field type is timestamp and needs conversion
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = {sourceObject}.{propName}.ToGrpcTimestamp(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = {sourceObject}.{propName}{comma}");
                }
            }
        }

        /// <summary>
        /// Check if a field is of timestamp type that needs conversion
        /// </summary>
        private bool IsTimestampField(ProtoField field)
        {
            return field.Type.Equals("Timestamp", StringComparison.OrdinalIgnoreCase) ||
                   field.Type.Equals("google.protobuf.Timestamp", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Get C# element type for repeated fields
        /// </summary>
        private string GetCsharpElementType(string protoType)
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
                _ => protoType // For custom types, use as-is
            };
        }

        /// <summary>
        /// Generate repeated field mapping with proper type conversion
        /// </summary>
        private void GenerateRepeatedFieldMapping(IndentStringBuilder sb, ProtoField field, string propName, 
            string sourceVar, string targetVar)
        {
            var elementType = GetCsharpElementType(field.Type);
            
            // Check if this is a primitive type
            if (IsPrimitiveType(field.Type))
            {
                // For primitive types, direct AddRange works
                sb.WriteLine($"{targetVar}.{propName}.AddRange({sourceVar}.{propName});");
            }
            else
            {
                // For custom types, generate conversion logic
                GenerateCustomTypeConversion(sb, field, propName, sourceVar, targetVar, ConversionDirection.DtoToGrpc);
            }
        }

        /// <summary>
        /// Generate repeated field mapping for client response (gRPC to DTO)
        /// </summary>
        private void GenerateClientRepeatedFieldMapping(IndentStringBuilder sb, ProtoField field, string propName, 
            string sourceVar, string targetVar)
        {
            // Check if this is a primitive type
            if (IsPrimitiveType(field.Type))
            {
                // For primitive types, direct ToList() works
                sb.WriteLine($"{targetVar}.{propName} = {sourceVar}.{propName}.ToList();");
            }
            else
            {
                // For custom types, generate conversion logic
                GenerateCustomTypeConversion(sb, field, propName, sourceVar, targetVar, ConversionDirection.GrpcToDto);
            }
        }

        /// <summary>
        /// Check if a type is a primitive protobuf type
        /// </summary>
        private bool IsPrimitiveType(string protoType)
        {
            return protoType switch
            {
                "int32" or "int64" or "uint32" or "uint64" or 
                "float" or "double" or "bool" or "string" or "bytes" => true,
                _ => false
            };
        }

        /// <summary>
        /// Conversion direction enumeration
        /// </summary>
        private enum ConversionDirection
        {
            DtoToGrpc,  // Convert from DTO to gRPC type
            GrpcToDto   // Convert from gRPC to DTO type
        }

        /// <summary>
        /// Generate custom type conversion for repeated fields
        /// </summary>
        private void GenerateCustomTypeConversion(IndentStringBuilder sb, ProtoField field, string propName, 
            string sourceVar, string targetVar, ConversionDirection direction)
        {
            var typeName = field.Type;
            
            switch (direction)
            {
                case ConversionDirection.DtoToGrpc:
                    // Convert from DTO to gRPC (e.g., CustomerInfoGrpcDto -> CustomerInfo)
                    sb.WriteLine($"{targetVar}.{propName}.AddRange({sourceVar}.{propName}.Select(dto => ConvertDtoTo{typeName}(dto)));");
                    break;
                    
                case ConversionDirection.GrpcToDto:
                    // Convert from gRPC to DTO (e.g., CustomerInfo -> CustomerInfoGrpcDto)
                    sb.WriteLine($"{targetVar}.{propName} = {sourceVar}.{propName}.Select(grpc => Convert{typeName}ToDto(grpc)).ToList();");
                    break;
            }
        }

        /// <summary>
        /// Generate conversion methods for server (gRPC to DTO and DTO to gRPC)
        /// </summary>
        private void GenerateServerConversionMethods(IndentStringBuilder sb, ProtoService svc, ProtoModel combineModel)
        {
            var customTypes = GetCustomTypesFromService(svc, combineModel);
            
            foreach (var customType in customTypes)
            {
                GenerateGrpcToDtoConversion(sb, customType, combineModel);
                GenerateDtoToGrpcConversion(sb, customType, combineModel);
            }
        }

        /// <summary>
        /// Generate conversion methods for client (DTO to gRPC and gRPC to DTO)
        /// </summary>
        private void GenerateClientConversionMethods(IndentStringBuilder sb, ProtoService svc, ProtoModel combineModel)
        {
            var customTypes = GetCustomTypesFromService(svc, combineModel);
            
            foreach (var customType in customTypes)
            {
                GenerateDtoToGrpcClientConversion(sb, customType, combineModel);
                GenerateGrpcToDtoClientConversion(sb, customType, combineModel);
            }
        }

        /// <summary>
        /// Get all custom types used in the service
        /// </summary>
        private HashSet<string> GetCustomTypesFromService(ProtoService svc, ProtoModel combineModel)
        {
            var customTypes = new HashSet<string>();
            
            foreach (var rpc in svc.Rpcs)
            {
                // Check request type
                var requestMessage = combineModel.FindMessage(rpc.RequestType);
                if (requestMessage != null)
                {
                    foreach (var field in requestMessage.Fields)
                    {
                        if (field.IsRepeated && !IsPrimitiveType(field.Type))
                        {
                            customTypes.Add(field.Type);
                        }
                    }
                }
                
                // Check response type
                var responseMessage = combineModel.FindMessage(rpc.ResponseType);
                if (responseMessage != null)
                {
                    foreach (var field in responseMessage.Fields)
                    {
                        if (field.IsRepeated && !IsPrimitiveType(field.Type))
                        {
                            customTypes.Add(field.Type);
                        }
                    }
                }
            }
            
            return customTypes;
        }

        /// <summary>
        /// Generate gRPC to DTO conversion method for server
        /// </summary>
        private void GenerateGrpcToDtoConversion(IndentStringBuilder sb, string typeName, ProtoModel combineModel)
        {
            var message = combineModel.FindMessage(typeName);
            if (message == null) return;

            sb.WriteLine($"private static {typeName}GrpcDto Convert{typeName}ToDto({typeName} grpc)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"return new {typeName}GrpcDto");
            sb.WriteLine("{");
            sb.Indent++;
            
            for (int i = 0; i < message.Fields.Count; i++)
            {
                var field = message.Fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < message.Fields.Count - 1 ? "," : "";
                
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = grpc.{propName}.ToDateTime(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = grpc.{propName}{comma}");
                }
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate DTO to gRPC conversion method for server
        /// </summary>
        private void GenerateDtoToGrpcConversion(IndentStringBuilder sb, string typeName, ProtoModel combineModel)
        {
            var message = combineModel.FindMessage(typeName);
            if (message == null) return;

            sb.WriteLine($"private static {typeName} ConvertDtoTo{typeName}({typeName}GrpcDto dto)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"return new {typeName}");
            sb.WriteLine("{");
            sb.Indent++;
            
            for (int i = 0; i < message.Fields.Count; i++)
            {
                var field = message.Fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < message.Fields.Count - 1 ? "," : "";
                
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = dto.{propName}.ToGrpcTimestamp(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = dto.{propName}{comma}");
                }
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate DTO to gRPC conversion method for client
        /// </summary>
        private void GenerateDtoToGrpcClientConversion(IndentStringBuilder sb, string typeName, ProtoModel combineModel)
        {
            var message = combineModel.FindMessage(typeName);
            if (message == null) return;

            sb.WriteLine($"private static {typeName} ConvertDtoTo{typeName}({typeName}GrpcDto dto)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"return new {typeName}");
            sb.WriteLine("{");
            sb.Indent++;
            
            for (int i = 0; i < message.Fields.Count; i++)
            {
                var field = message.Fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < message.Fields.Count - 1 ? "," : "";
                
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = dto.{propName}.ToGrpcTimestamp(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = dto.{propName}{comma}");
                }
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
        }

        /// <summary>
        /// Generate repeated field mapping for server request (gRPC to DTO)
        /// </summary>
        private void GenerateServerRequestRepeatedFieldMapping(IndentStringBuilder sb, ProtoField field, string propName, 
            string sourceVar, string targetVar)
        {
            // Check if this is a primitive type
            if (IsPrimitiveType(field.Type))
            {
                // For primitive types, direct ToList() works
                sb.WriteLine($"{targetVar}.{propName} = {sourceVar}.{propName}.ToList();");
            }
            else
            {
                // For custom types, generate conversion logic (gRPC to DTO)
                GenerateCustomTypeConversion(sb, field, propName, sourceVar, targetVar, ConversionDirection.GrpcToDto);
            }
        }

        /// <summary>
        /// Generate gRPC to DTO conversion method for client
        /// </summary>
        private void GenerateGrpcToDtoClientConversion(IndentStringBuilder sb, string typeName, ProtoModel combineModel)
        {
            var message = combineModel.FindMessage(typeName);
            if (message == null) return;

            sb.WriteLine($"private static {typeName}GrpcDto Convert{typeName}ToDto({typeName} grpc)");
            sb.WriteLine("{");
            sb.Indent++;
            sb.WriteLine($"return new {typeName}GrpcDto");
            sb.WriteLine("{");
            sb.Indent++;
            
            for (int i = 0; i < message.Fields.Count; i++)
            {
                var field = message.Fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < message.Fields.Count - 1 ? "," : "";
                
                if (IsTimestampField(field))
                {
                    sb.WriteLine($"{propName} = grpc.{propName}.ToDateTime(){comma}");
                }
                else
                {
                    sb.WriteLine($"{propName} = grpc.{propName}{comma}");
                }
            }
            
            sb.Indent--;
            sb.WriteLine("};");
            sb.Indent--;
            sb.WriteLine("}");
            sb.WriteLine();
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

    public class ProtoFileInfo
    {
        public string Path { get; set; }
        public string Content { get; set; }

        public string GetProtoFileName()
        {
            return System.IO.Path.GetFileNameWithoutExtension(Path);
        }
    }

    /// <summary>
    /// Service method information for code generation
    /// </summary>
    internal class ServiceMethodInfo
    {
        public string RequestFullType { get; set; } = string.Empty;
        public string ResponseFullType { get; set; } = string.Empty;
        public string RequestType { get; set; } = string.Empty;
        public string ResponseType { get; set; } = string.Empty;
        public bool IsNullRequest { get; set; }
        public bool IsVoidResponse { get; set; }
    }

    /// <summary>
    /// Interface method signature information
    /// </summary>
    internal class InterfaceMethodSignature
    {
        public string Parameters { get; set; } = string.Empty;
        public string ReturnType { get; set; } = string.Empty;
    }

    /// <summary>
    /// Client method information for code generation
    /// </summary>
    internal class ClientMethodInfo
    {
        public bool IsNullRequest { get; set; }
        public bool IsVoidResponse { get; set; }
    }

    /// <summary>
    /// Client interface method signature information
    /// </summary>
    internal class ClientInterfaceMethodSignature
    {
        public string Parameters { get; set; } = string.Empty;
        public string ReturnType { get; set; } = string.Empty;
    }

    /// <summary>
    /// Message property information for code generation
    /// </summary>
    internal class MessagePropertyInfo
    {
        public string Type { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public bool IsOptional { get; set; }
    }

    /// <summary>
    /// Collection of proto elements from all proto files
    /// </summary>
    internal class ProtoElementCollection
    {
        public List<ProtoMessage> Messages { get; set; } = new List<ProtoMessage>();
        public List<ProtoEnum> Enums { get; set; } = new List<ProtoEnum>();
        public List<ProtoService> Services { get; set; } = new List<ProtoService>();
    }
}

