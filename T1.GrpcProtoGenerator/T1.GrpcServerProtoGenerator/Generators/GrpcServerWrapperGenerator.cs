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

            context.RegisterSourceOutput(protoFilesWithContent, (spc, protoInfo) =>
            {
                var model = ProtoParser.ParseProtoText(protoInfo.Content);
                var protoFileName = protoInfo.GetProtoFileName();

                AddGeneratedSourceFile(spc, GenerateWrapperGrpcMessageSource(model), $"Generated_{protoFileName}_messages.cs");
                AddGeneratedSourceFile(spc, GenerateWrapperServerSource(model), $"Generated_{protoFileName}_server.cs");
                AddGeneratedSourceFile(spc, GenerateWrapperClientSource(model), $"Generated_{protoFileName}_client.cs");
            });
        }

        private static void AddGeneratedSourceFile(SourceProductionContext spc, string messagesSource, string sourceFileName)
        {
            spc.AddSource(sourceFileName, SourceText.From(messagesSource, Encoding.UTF8));
        }

        private string GenerateWrapperGrpcMessageSource(ProtoModel model)
        {
            var sb = new StringBuilder();
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine();
            
            var targetNamespace = model.GetTatgetNamespace();
            sb.AppendLine($"namespace {targetNamespace}");
            sb.AppendLine("{");

            foreach (var msg in model.Messages)
            {
                sb.AppendLine($"    public class {msg.Name}GrpcMessage");
                sb.AppendLine("    {");
                foreach (var f in msg.Fields)
                {
                    var baseType = MapProtoCTypeToCSharp(f.Type);
                    var csType = f.IsRepeated ? $"List<{baseType}>" : baseType;
                    sb.AppendLine($"        public {csType} {char.ToUpper(f.Name[0]) + f.Name.Substring(1)} {{ get; set; }}");
                }
                sb.AppendLine("    }");
                sb.AppendLine();
            }

            foreach (var e in model.Enums)
            {
                sb.AppendLine($"    public enum {e.Name}");
                sb.AppendLine("    {");
                foreach (var val in e.Values)
                    sb.AppendLine($"        {val.Name} = {val.Value},");
                sb.AppendLine("    }");
                sb.AppendLine();
            }

            sb.AppendLine("}");
            return sb.ToString();
        }

        private string GenerateWrapperClientSource(ProtoModel model)
        {
            var sb = new StringBuilder();
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using System.Threading;");
            sb.AppendLine("using System.Threading.Tasks;");
            sb.AppendLine("using Grpc.Core;");
            sb.AppendLine();
            
            var targetNamespace = model.GetTatgetNamespace();
            sb.AppendLine($"namespace {targetNamespace}");
            sb.AppendLine("{");

            foreach (var svc in model.Services)
            {
                var originalNamespace = !string.IsNullOrEmpty(model.CsharpNamespace) ? model.CsharpNamespace : "Generated";
                
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
                    sb.AppendLine($"        public async Task<{rpc.ResponseType}GrpcMessage> {rpc.Name}Async({rpc.RequestType}GrpcMessage request, CancellationToken cancellationToken = default)");
                    sb.AppendLine("        {");
                    sb.AppendLine($"            var grpcReq = new {originalNamespace}.{rpc.RequestType}();");
                    
                    var requestMessage = model.FindMessage(rpc.RequestType);
                    if (requestMessage != null)
                    {
                        foreach (var field in requestMessage.Fields)
                        {
                            var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                            sb.AppendLine($"            grpcReq.{propName} = request.{propName};");
                        }
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
                    
                    sb.AppendLine("            return dto;");
                    sb.AppendLine("        }");
                    sb.AppendLine();
                }
                sb.AppendLine("    }");
            }

            sb.AppendLine("}");
            return sb.ToString();
        }

        private string GenerateWrapperServerSource(ProtoModel model)
        {
            var sb = new StringBuilder();
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using System.Threading;");
            sb.AppendLine("using System.Threading.Tasks;");
            sb.AppendLine("using Grpc.Core;");
            sb.AppendLine("using Microsoft.Extensions.Logging;");
            sb.AppendLine();
            
            var targetNamespace = model.GetTatgetNamespace();
            sb.AppendLine($"namespace {targetNamespace}");
            sb.AppendLine("{");

            foreach (var svc in model.Services)
            {
                var originalNamespace = !string.IsNullOrEmpty(model.CsharpNamespace) ? model.CsharpNamespace : "Generated";
                
                var serviceInterface = $"I{svc.Name}GrpcService";
                sb.AppendLine($"    public interface {serviceInterface}");
                sb.AppendLine("    {");
                foreach (var rpc in svc.Rpcs)
                {
                    sb.AppendLine($"        Task<{rpc.ResponseType}GrpcMessage> {rpc.Name}({rpc.RequestType}GrpcMessage request, ServerCallContext context);");
                }
                sb.AppendLine("    }");
                sb.AppendLine();

                var serviceClass = $"{svc.Name}GrpcNativeService";
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
                    sb.AppendLine($"        public override async Task<{originalNamespace}.{rpc.ResponseType}> {rpc.Name}({originalNamespace}.{rpc.RequestType} request, ServerCallContext context)");
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
                    
                    sb.AppendLine($"            var dtoResponse = await _instance.{rpc.Name}(dtoRequest, context);");
                    sb.AppendLine($"            var grpcResponse = new {originalNamespace}.{rpc.ResponseType}();");
                    
                    var responseMessage = model.FindMessage(rpc.ResponseType);
                    if (responseMessage != null)
                    {
                        foreach (var field in responseMessage.Fields)
                        {
                            var propName = char.ToUpper(field.Name[0]) + field.Name.Substring(1);
                            sb.AppendLine($"            grpcResponse.{propName} = dtoResponse.{propName};");
                        }
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
}
