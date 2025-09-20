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
                var logger = InitializeLogger(spc);
                logger.LogWarning($"Starting source generation for {allProtos.Length} proto files");
                
                var combinedModel = CreateCombinedModel(allProtos);
                
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

        /// <summary>
        /// Create a combined model with unique messages and enums from all proto files
        /// </summary>
        private ProtoModel CreateCombinedModel(ImmutableArray<ProtoFileInfo> allProtos)
        {
            var combinedModel = new ProtoModel();
            
            // Collect all messages and enums from all proto files
            var allMessages = new List<ProtoMessage>();
            var allEnums = new List<ProtoEnum>();
            var allSvcList = new List<ProtoService>();
            
            foreach (var protoInfo in allProtos)
            {
                var model = ProtoParser.ParseProtoText(protoInfo.Content, protoInfo.Path);
                allMessages.AddRange(model.Messages);
                allEnums.AddRange(model.Enums);
                allSvcList.AddRange(model.Services);
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

            foreach (var svc in allSvcList)
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


        private string GenerateWrapperGrpcMessageSource(ProtoMessage messageModel, ProtoModel combinedModel)
        {
            var sb = new StringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            sb.AppendLine($"namespace {messageModel.CsharpNamespace}");
            sb.AppendLine("{");

            GenerateSingleMessageClass(sb, messageModel, combinedModel);

            sb.AppendLine("}");
            sb.AppendLine();

            return sb.ToString();
        }

        private string GenerateWrapperGrpcEnumSource(ProtoEnum enumModel)
        {
            var sb = new StringBuilder();
            
            // Generate using statements
            GenerateBasicUsingStatements(sb);
            
            sb.AppendLine($"namespace {enumModel.CsharpNamespace}");
            sb.AppendLine("{");

            GenerateSingleEnumClass(sb, enumModel);

            sb.AppendLine("}");
            sb.AppendLine();

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
        /// Generate a single message class with its properties
        /// </summary>
        private void GenerateSingleMessageClass(StringBuilder sb, ProtoMessage msg, ProtoModel combineModel)
        {
            // Skip generating DTO for Google Void/Null types as they correspond to no C# class needed
            if (msg.Name.Equals("Void", StringComparison.OrdinalIgnoreCase) || 
                msg.Name.Equals("Null", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }
            
            sb.AppendLine($"    public class {msg.GetCsharpTypeName()}");
            sb.AppendLine("    {");
            
            foreach (var field in msg.Fields)
            {
                var baseType = MapProtoCTypeToCSharp(field.Type, combineModel);
                if (field.IsOption)
                {
                    baseType += "?";
                }
                var csType = field.IsRepeated ? $"List<{baseType}>" : baseType;
                var propertyName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                if (field.IsOption)
                {
                    sb.AppendLine($"        public {csType} {propertyName} {{ get; set; }}");
                }
                else
                {
                    sb.AppendLine($"        public required {csType} {propertyName} {{ get; set; }}");
                }
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

        private string GenerateWrapperClientSource(ProtoModel model, ProtoModel combinedModel)
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
                    GenerateClientWrapper(sb, svc, combinedModel);
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
                // Handle request parameter - skip parameter if request type is "Null"
                string parameterPart = "";
                if (!rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase))
                {
                    parameterPart = $"{rpc.RequestType}GrpcDto request, ";
                }
                
                // Handle return type - use Task instead of Task<T> if response type is "Void"
                string returnType;
                if (rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase))
                {
                    returnType = "Task";
                }
                else
                {
                    returnType = $"Task<{rpc.ResponseType}GrpcDto>";
                }
                
                sb.AppendLine($"        {returnType} {rpc.Name}Async({parameterPart}CancellationToken cancellationToken = default);");
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate client wrapper class for a given service
        /// </summary>
        private void GenerateClientWrapper(StringBuilder sb, ProtoService svc, ProtoModel combineModel)
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
                GenerateClientMethod(sb, rpc, combineModel);
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate a single client method implementation
        /// </summary>
        private void GenerateClientMethod(StringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var clientMethodInfo = GetClientMethodInfo(rpc);
            
            // Generate method signature
            GenerateClientMethodSignature(sb, rpc, clientMethodInfo);
            
            // Generate method body
            sb.AppendLine("        {");
            
            // Generate request mapping
            GenerateClientRequestMapping(sb, rpc, clientMethodInfo, combineModel);
            
            // Generate service call and response handling
            GenerateClientServiceCallAndResponse(sb, rpc, clientMethodInfo, combineModel);
            
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        /// <summary>
        /// Get client method information including type flags
        /// </summary>
        private ClientMethodInfo GetClientMethodInfo(ProtoRpc rpc)
        {
            return new ClientMethodInfo
            {
                IsNullRequest = rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase),
                IsVoidResponse = rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase)
            };
        }

        /// <summary>
        /// Generate client method signature
        /// </summary>
        private void GenerateClientMethodSignature(StringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo)
        {
            var parameterPart = methodInfo.IsNullRequest ? "" : $"{rpc.RequestType}GrpcDto request, ";
            var returnType = methodInfo.IsVoidResponse ? "Task" : $"Task<{rpc.ResponseType}GrpcDto>";
            
            sb.AppendLine($"        public async {returnType} {rpc.Name}Async({parameterPart}CancellationToken cancellationToken = default)");
        }

        /// <summary>
        /// Generate client request mapping from DTO to gRPC
        /// </summary>
        private void GenerateClientRequestMapping(StringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (!methodInfo.IsNullRequest)
            {
                GenerateClientRequestObjectMapping(sb, rpc, combineModel);
            }
            else
            {
                GenerateClientNullRequestMapping(sb, rpc);
            }
        }

        /// <summary>
        /// Generate client request object mapping for non-null requests
        /// </summary>
        private void GenerateClientRequestObjectMapping(StringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var requestMessage = combineModel.FindMessage(rpc.RequestType);
            sb.AppendLine($"            var grpcReq = new {rpc.RequestType}");
            sb.AppendLine("            {");
            
            GenerateFieldMappings(sb, requestMessage.Fields, "request");
            
            sb.AppendLine("            };");
        }

        /// <summary>
        /// Generate client request mapping for null requests
        /// </summary>
        private void GenerateClientNullRequestMapping(StringBuilder sb, ProtoRpc rpc)
        {
            sb.AppendLine($"            var grpcReq = new {rpc.RequestType}();");
        }

        /// <summary>
        /// Generate client service call and response handling
        /// </summary>
        private void GenerateClientServiceCallAndResponse(StringBuilder sb, ProtoRpc rpc, ClientMethodInfo methodInfo, ProtoModel combineModel)
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
        private void GenerateClientVoidServiceCall(StringBuilder sb, ProtoRpc rpc)
        {
            sb.AppendLine($"            await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
        }

        /// <summary>
        /// Generate client service call for normal response methods
        /// </summary>
        private void GenerateClientNormalServiceCall(StringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            sb.AppendLine($"            var grpcResp = await _inner.{rpc.Name}Async(grpcReq, cancellationToken: cancellationToken);");
            
            // Generate response mapping
            GenerateClientResponseMapping(sb, rpc, combineModel);
        }

        /// <summary>
        /// Generate client response mapping from gRPC to DTO
        /// </summary>
        private void GenerateClientResponseMapping(StringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var responseMessage = combineModel.FindMessage(rpc.ResponseType);
            sb.AppendLine($"            var dto = new {rpc.ResponseType}GrpcDto");
            sb.AppendLine("            {");
            
            GenerateFieldMappings(sb, responseMessage.Fields, "grpcResp");
            
            sb.AppendLine("            };");
            sb.AppendLine("            return dto;");
        }

        private string GenerateWrapperServerSource(ProtoModel model, ProtoModel combineModel)
        {
            if (!ValidateServerSourceGeneration(model))
            {
                return string.Empty;
            }
            
            var sb = new StringBuilder();
            
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
        private void SetupServerSourceUsingStatements(StringBuilder sb, ProtoModel combineModel)
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
        private void GenerateNamespaceBlocks(StringBuilder sb, 
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
        private void GenerateNamespaceDeclaration(StringBuilder sb, string namespaceValue)
        {
            sb.AppendLine($"namespace {namespaceValue}");
            sb.AppendLine("{");
        }

        /// <summary>
        /// Generate all services within a namespace
        /// </summary>
        private void GenerateServicesInNamespace(StringBuilder sb, 
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
        /// Generate namespace closing brace
        /// </summary>
        private void GenerateNamespaceClosing(StringBuilder sb)
        {
            sb.AppendLine("}");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate service interface for a given service
        /// </summary>
        private void GenerateServiceInterface(StringBuilder sb, ProtoService svc, ProtoModel messagesModel)
        {
            GenerateInterfaceDeclaration(sb, svc);
            GenerateInterfaceMethods(sb, svc, messagesModel);
            GenerateInterfaceClosing(sb);
        }

        /// <summary>
        /// Generate interface declaration and opening brace
        /// </summary>
        private void GenerateInterfaceDeclaration(StringBuilder sb, ProtoService svc)
        {
            var serviceInterface = $"I{svc.Name}GrpcService";
            sb.AppendLine($"    public interface {serviceInterface}");
            sb.AppendLine("    {");
        }

        /// <summary>
        /// Generate all interface methods for the service
        /// </summary>
        private void GenerateInterfaceMethods(StringBuilder sb, ProtoService svc, ProtoModel messagesModel)
        {
            foreach (var rpc in svc.Rpcs)
            {
                GenerateSingleInterfaceMethod(sb, rpc, messagesModel);
            }
        }

        /// <summary>
        /// Generate a single interface method signature
        /// </summary>
        private void GenerateSingleInterfaceMethod(StringBuilder sb, ProtoRpc rpc, ProtoModel messagesModel)
        {
            var methodSignature = CreateInterfaceMethodSignature(rpc, messagesModel);
            sb.AppendLine($"        {methodSignature.ReturnType} {rpc.Name}({methodSignature.Parameters});");
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
            // Skip parameter if request type is "Null"
            if (rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase))
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
            // Use Task instead of Task<T> if response type is "Void"
            if (rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase))
            {
                return "Task";
            }
            
            return $"Task<{rpcResponseType}>";
        }

        /// <summary>
        /// Generate interface closing brace
        /// </summary>
        private void GenerateInterfaceClosing(StringBuilder sb)
        {
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate service implementation class for a given service
        /// </summary>
        private void GenerateServiceImplementation(StringBuilder sb, ProtoService svc,
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
                GenerateServiceMethod(sb, rpc, combineModel);
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        /// <summary>
        /// Generate a single service method implementation
        /// </summary>
        private void GenerateServiceMethod(StringBuilder sb, ProtoRpc rpc, ProtoModel combineModel)
        {
            var methodInfo = GetServiceMethodInfo(rpc, combineModel);
            
            // Generate method signature
            GenerateServiceMethodSignature(sb, rpc, methodInfo);
            
            // Generate method body
            sb.AppendLine("        {");
            
            // Generate request mapping
            GenerateRequestMapping(sb, rpc, methodInfo, combineModel);
            
            // Generate service call and response handling
            GenerateServiceCallAndResponse(sb, rpc, methodInfo, combineModel);
            
            sb.AppendLine("        }");
            sb.AppendLine();
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
                IsNullRequest = rpc.RequestType.Equals("Null", StringComparison.OrdinalIgnoreCase),
                IsVoidResponse = rpc.ResponseType.Equals("Void", StringComparison.OrdinalIgnoreCase)
            };
        }

        /// <summary>
        /// Generate service method signature
        /// </summary>
        private void GenerateServiceMethodSignature(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo)
        {
            sb.AppendLine($"        public override async Task<{methodInfo.ResponseFullType}> {rpc.Name}({methodInfo.RequestFullType} request, ServerCallContext context)");
        }

        /// <summary>
        /// Generate request parameter mapping from gRPC to DTO
        /// </summary>
        private void GenerateRequestMapping(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            if (!methodInfo.IsNullRequest)
            {
                var requestMessage = combineModel.FindMessage(rpc.RequestType);
                sb.AppendLine($"            var dtoRequest = new {methodInfo.RequestType}");
                sb.AppendLine("            {");
                
                GenerateFieldMappings(sb, requestMessage.Fields, "request");
                
                sb.AppendLine("            };");
            }
        }

        /// <summary>
        /// Generate service call and response handling
        /// </summary>
        private void GenerateServiceCallAndResponse(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
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
        private void GenerateVoidServiceCall(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo)
        {
            // Call service method without expecting return value
            if (methodInfo.IsNullRequest)
            {
                sb.AppendLine($"            await _instance.{rpc.Name}();");
            }
            else
            {
                sb.AppendLine($"            await _instance.{rpc.Name}(dtoRequest);");
            }
            
            // Return empty response instance
            sb.AppendLine($"            return new {methodInfo.ResponseFullType}();");
        }

        /// <summary>
        /// Generate service call for normal response methods
        /// </summary>
        private void GenerateNormalServiceCall(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            // Call service method and get return value
            if (methodInfo.IsNullRequest)
            {
                sb.AppendLine($"            var dtoResponse = await _instance.{rpc.Name}();");
            }
            else
            {
                sb.AppendLine($"            var dtoResponse = await _instance.{rpc.Name}(dtoRequest);");
            }
            
            // Generate response mapping
            GenerateResponseMapping(sb, rpc, methodInfo, combineModel);
        }

        /// <summary>
        /// Generate response mapping from DTO to gRPC
        /// </summary>
        private void GenerateResponseMapping(StringBuilder sb, ProtoRpc rpc, ServiceMethodInfo methodInfo, ProtoModel combineModel)
        {
            var responseMessage = combineModel.FindMessage(rpc.ResponseType);
            sb.AppendLine($"            var grpcResponse = new {methodInfo.ResponseFullType}");
            sb.AppendLine("            {");
            
            GenerateFieldMappings(sb, responseMessage.Fields, "dtoResponse");
            
            sb.AppendLine("            };");
            sb.AppendLine("            return grpcResponse;");
        }

        /// <summary>
        /// Generate field mappings for object initializer
        /// </summary>
        private void GenerateFieldMappings(StringBuilder sb, List<ProtoField> fields, string sourceObject)
        {
            for (int i = 0; i < fields.Count; i++)
            {
                var field = fields[i];
                var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                var comma = i < fields.Count - 1 ? "," : "";
                sb.AppendLine($"                {propName} = {sourceObject}.{propName}{comma}");
            }
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
}
