using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace T1.GrpcSourceGenerator
{
    [Generator]
    public class GrpcServiceSourceGenerator : ISourceGenerator
    {
        private const string AttributeFullName = "T1.GrpcSourceGenerator.GenerateGrpcServiceAttribute";

        public void Initialize(GeneratorInitializationContext context)
        {
            // 註冊語法接收器
            context.RegisterForSyntaxNotifications(() => new SyntaxReceiver());
        }

        public void Execute(GeneratorExecutionContext context)
        {
            // 處理候選類
            if (!(context.SyntaxContextReceiver is SyntaxReceiver receiver) || receiver.CandidateClasses.Count == 0)
                return;

            // 獲取 Attribute 符號
            INamedTypeSymbol attributeSymbol = context.Compilation.GetTypeByMetadataName(AttributeFullName);
            if (attributeSymbol == null)
            {
                // 如果找不到 Attribute 類型，生成一個
                GenerateAttributeClass(context);
                return; // 在下一個編譯循環中再處理
            }

            // 處理每個標記了 Attribute 的類
            foreach (var candidateClass in receiver.CandidateClasses)
            {
                ProcessClass(context, candidateClass.ClassSymbol, candidateClass.InterfaceSymbol);
            }
        }

        private void GenerateAttributeClass(GeneratorExecutionContext context)
        {
            string attributeSource = @"
using System;

namespace T1.GrpcSourceGenerator
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class GenerateGrpcServiceAttribute : Attribute
    {
        public Type InterfaceType { get; }

        public GenerateGrpcServiceAttribute(Type interfaceType)
        {
            InterfaceType = interfaceType ?? throw new ArgumentNullException(nameof(interfaceType));
        }
    }
}";
            context.AddSource("GenerateGrpcServiceAttribute.g.cs", SourceText.From(attributeSource, Encoding.UTF8));
        }

        private void ProcessClass(GeneratorExecutionContext context, INamedTypeSymbol classSymbol, INamedTypeSymbol interfaceSymbol)
        {
            string className = classSymbol.Name;
            string interfaceName = interfaceSymbol.Name;
            
            // 從接口名稱得到服務名稱 (去掉開頭的 'I')
            string serviceName = interfaceName.StartsWith("I") ? interfaceName.Substring(1) : interfaceName;

            // 生成 .proto 文件內容 (僅作為註釋添加，不生成實際文件)
            string protoContent = GenerateProtoFileContent(interfaceSymbol, serviceName);
            context.AddSource($"{serviceName}Messages.proto.g.cs", SourceText.From(
                $"// 以下是自動生成的 .proto 文件內容\r\n" +
                $"// 請將此內容複製到一個 .proto 文件中，並使用 Grpc.Tools 編譯\r\n" +
                $"/*\r\n{protoContent}\r\n*/", 
                Encoding.UTF8));

            // 生成 gRPC 服務實現類
            GenerateGrpcServiceClass(context, classSymbol, interfaceSymbol, serviceName);
        }

        private string GenerateProtoFileContent(INamedTypeSymbol interfaceSymbol, string serviceName)
        {
            StringBuilder protoBuilder = new StringBuilder();

            // Proto 文件頭
            protoBuilder.AppendLine("syntax = \"proto3\";");
            protoBuilder.AppendLine();

            // 命名空間/包名
            string packageName = interfaceSymbol.ContainingNamespace.ToDisplayString().ToLowerInvariant();
            protoBuilder.AppendLine($"package {packageName};");
            protoBuilder.AppendLine();

            // 導入必要的 proto 定義
            protoBuilder.AppendLine("import \"google/protobuf/timestamp.proto\";");
            protoBuilder.AppendLine("import \"google/protobuf/empty.proto\";");
            protoBuilder.AppendLine();

            // 用於跟踪已經生成的消息類型
            HashSet<string> generatedMessages = new HashSet<string>();

            // 為每個方法生成消息定義
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    GenerateMessagesForMethodProto(protoBuilder, methodSymbol, generatedMessages);
                }
            }

            // 生成服務定義
            protoBuilder.AppendLine($"service {serviceName} {{");
            
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    string methodName = methodSymbol.Name;
                    string requestMessageName = $"{methodName}RequestMessage";
                    string replyMessageName = $"{methodName}ReplyMessage";
                    
                    protoBuilder.AppendLine($"  rpc {methodName} ({requestMessageName}) returns ({replyMessageName});");
                }
            }
            
            protoBuilder.AppendLine("}");

            return protoBuilder.ToString();
        }

        private void GenerateMessagesForMethodProto(StringBuilder protoBuilder, IMethodSymbol methodSymbol, HashSet<string> generatedMessages)
        {
            string methodName = methodSymbol.Name;
            
            // 請求消息
            string requestMessageName = $"{methodName}RequestMessage";
            if (!generatedMessages.Contains(requestMessageName))
            {
                protoBuilder.AppendLine($"message {requestMessageName} {{");
                
                int fieldIndex = 1;
                foreach (var parameter in methodSymbol.Parameters)
                {
                    string protoType = GetProtoType(parameter.Type);
                    string fieldName = ToCamelCase(parameter.Name);
                    protoBuilder.AppendLine($"  {protoType} {fieldName} = {fieldIndex};");
                    fieldIndex++;
                }
                
                protoBuilder.AppendLine("}");
                protoBuilder.AppendLine();
                
                generatedMessages.Add(requestMessageName);
            }
            
            // 回應消息
            string replyMessageName = $"{methodName}ReplyMessage";
            if (!generatedMessages.Contains(replyMessageName))
            {
                protoBuilder.AppendLine($"message {replyMessageName} {{");
                
                ITypeSymbol returnType = GetActualReturnType(methodSymbol);
                if (returnType != null && returnType.SpecialType != SpecialType.System_Void)
                {
                    string protoType = GetProtoType(returnType);
                    protoBuilder.AppendLine($"  {protoType} result = 1;");
                }
                
                protoBuilder.AppendLine("}");
                protoBuilder.AppendLine();
                
                generatedMessages.Add(replyMessageName);
            }
        }

        private void GenerateGrpcServiceClass(GeneratorExecutionContext context, INamedTypeSymbol classSymbol, 
                                             INamedTypeSymbol interfaceSymbol, string serviceName)
        {
            StringBuilder serviceBuilder = new StringBuilder();
            
            // 添加必要的 using 語句
            serviceBuilder.AppendLine("// <auto-generated />");
            serviceBuilder.AppendLine("#pragma warning disable CS1591, CS0612, CS3021, IDE1006");
            serviceBuilder.AppendLine("#nullable enable");
            serviceBuilder.AppendLine();
            serviceBuilder.AppendLine("using System;");
            serviceBuilder.AppendLine("using System.Threading.Tasks;");
            serviceBuilder.AppendLine("using Grpc.Core;");
            serviceBuilder.AppendLine($"using {interfaceSymbol.ContainingNamespace.ToDisplayString()};");
            serviceBuilder.AppendLine();
            
            // 命名空間
            string namespaceName = classSymbol.ContainingNamespace.ToDisplayString();
            serviceBuilder.AppendLine($"namespace {namespaceName}");
            serviceBuilder.AppendLine("{");
            
            // 生成空的消息類型 (用於編譯通過，實際使用時需要替換為 protoc 生成的類型)
            // 這裡僅為示例，實際應用中應從 proto 文件生成真正的消息類型
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    string methodName = methodSymbol.Name;
                    string requestMessageName = $"{methodName}RequestMessage";
                    string replyMessageName = $"{methodName}ReplyMessage";

                    // 生成示例請求消息類型
                    serviceBuilder.AppendLine($"    // 請替換為實際由 protoc 生成的消息類型");
                    serviceBuilder.AppendLine($"    public class {requestMessageName}");
                    serviceBuilder.AppendLine("    {");
                    foreach (var parameter in methodSymbol.Parameters)
                    {
                        string propertyName = ToPascalCase(parameter.Name);
                        string typeName = parameter.Type.ToDisplayString();
                        serviceBuilder.AppendLine($"        public {typeName} {propertyName} {{ get; set; }}");
                    }
                    serviceBuilder.AppendLine("    }");
                    serviceBuilder.AppendLine();

                    // 生成示例回應消息類型
                    serviceBuilder.AppendLine($"    public class {replyMessageName}");
                    serviceBuilder.AppendLine("    {");
                    ITypeSymbol returnType = GetActualReturnType(methodSymbol);
                    if (returnType != null && returnType.SpecialType != SpecialType.System_Void)
                    {
                        string typeName = returnType.ToDisplayString();
                        serviceBuilder.AppendLine($"        public {typeName} Result {{ get; set; }}");
                    }
                    serviceBuilder.AppendLine("    }");
                    serviceBuilder.AppendLine();
                }
            }

            // 生成基類 (僅為示例，實際應用中應使用 protoc 生成的基類)
            serviceBuilder.AppendLine($"    // 請替換為實際由 protoc 生成的基類");
            serviceBuilder.AppendLine($"    public abstract class {serviceName}Base");
            serviceBuilder.AppendLine("    {");
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    string methodName = methodSymbol.Name;
                    string requestMessageName = $"{methodName}RequestMessage";
                    string replyMessageName = $"{methodName}ReplyMessage";
                    
                    serviceBuilder.AppendLine($"        public virtual Task<{replyMessageName}> {methodName}({requestMessageName} request, ServerCallContext context) => Task.FromResult(new {replyMessageName}());");
                }
            }
            serviceBuilder.AppendLine("    }");
            serviceBuilder.AppendLine();
            
            // 生成服務實現類
            serviceBuilder.AppendLine($"    /// <summary>");
            serviceBuilder.AppendLine($"    /// 自動生成的 gRPC 服務實現，基於 {interfaceSymbol.Name}");
            serviceBuilder.AppendLine($"    /// </summary>");
            serviceBuilder.AppendLine($"    public class {serviceName}GrpcService : {serviceName}Base");
            serviceBuilder.AppendLine("    {");
            
            // 字段和構造函數
            serviceBuilder.AppendLine($"        private readonly {interfaceSymbol.Name} _{ToCamelCase(serviceName)};");
            serviceBuilder.AppendLine();
            serviceBuilder.AppendLine($"        public {serviceName}GrpcService({interfaceSymbol.Name} {ToCamelCase(serviceName)})");
            serviceBuilder.AppendLine("        {");
            serviceBuilder.AppendLine($"            _{ToCamelCase(serviceName)} = {ToCamelCase(serviceName)};");
            serviceBuilder.AppendLine("        }");
            serviceBuilder.AppendLine();
            
            // 生成每個方法的實現
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    GenerateMethodImplementation(serviceBuilder, methodSymbol, serviceName);
                }
            }
            
            serviceBuilder.AppendLine("    }");
            serviceBuilder.AppendLine("}");
            serviceBuilder.AppendLine("#pragma warning restore CS1591, CS0612, CS3021, IDE1006");
            serviceBuilder.AppendLine("#nullable restore");
            
            // 添加生成的服務類
            context.AddSource($"{serviceName}GrpcService.g.cs", SourceText.From(serviceBuilder.ToString(), Encoding.UTF8));
        }

        private void GenerateMethodImplementation(StringBuilder sb, IMethodSymbol methodSymbol, string serviceName)
        {
            string methodName = methodSymbol.Name;
            string requestMessageName = $"{methodName}RequestMessage";
            string replyMessageName = $"{methodName}ReplyMessage";
            
            // 方法簽名
            sb.AppendLine($"        public override async Task<{replyMessageName}> {methodName}({requestMessageName} request, ServerCallContext context)");
            sb.AppendLine("        {");
            
            // 參數轉換
            List<string> argList = new List<string>();
            foreach (var parameter in methodSymbol.Parameters)
            {
                string paramName = parameter.Name;
                string requestProp = ToPascalCase(paramName);
                sb.AppendLine($"            var {paramName} = request.{requestProp};");
                argList.Add(paramName);
            }
            
            string arguments = string.Join(", ", argList);
            
            // 調用接口方法
            bool isAsync = IsAsyncMethod(methodSymbol);
            if (isAsync)
            {
                sb.AppendLine($"            var result = await _{ToCamelCase(serviceName)}.{methodName}({arguments});");
            }
            else
            {
                sb.AppendLine($"            var result = _{ToCamelCase(serviceName)}.{methodName}({arguments});");
            }
            
            // 返回結果
            ITypeSymbol returnType = GetActualReturnType(methodSymbol);
            if (returnType != null && returnType.SpecialType != SpecialType.System_Void)
            {
                sb.AppendLine($"            return new {replyMessageName} {{ Result = result }};");
            }
            else
            {
                sb.AppendLine($"            return new {replyMessageName}();");
            }
            
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        private bool IsAsyncMethod(IMethodSymbol methodSymbol)
        {
            return methodSymbol.ReturnType is INamedTypeSymbol namedType &&
                   namedType.Name == "Task" &&
                   namedType.ContainingNamespace.ToDisplayString() == "System.Threading.Tasks";
        }

        private ITypeSymbol GetActualReturnType(IMethodSymbol methodSymbol)
        {
            if (methodSymbol.ReturnType is INamedTypeSymbol namedType && 
                namedType.Name == "Task" && 
                namedType.ContainingNamespace.ToDisplayString() == "System.Threading.Tasks")
            {
                if (namedType.IsGenericType && namedType.TypeArguments.Length > 0)
                {
                    return namedType.TypeArguments[0];
                }
                return null; // Task 無泛型參數，即 void
            }
            return methodSymbol.ReturnType;
        }

        private string GetProtoType(ITypeSymbol type)
        {
            switch (type.SpecialType)
            {
                case SpecialType.System_Boolean:
                    return "bool";
                case SpecialType.System_SByte:
                case SpecialType.System_Byte:
                    return "bytes";
                case SpecialType.System_Int16:
                case SpecialType.System_UInt16:
                case SpecialType.System_Int32:
                    return "int32";
                case SpecialType.System_UInt32:
                case SpecialType.System_Int64:
                    return "int64";
                case SpecialType.System_UInt64:
                    return "uint64";
                case SpecialType.System_Single:
                    return "float";
                case SpecialType.System_Double:
                    return "double";
                case SpecialType.System_String:
                    return "string";
                default:
                    if (type.Name == "Guid")
                        return "string";
                    if (type.Name == "DateTime" || type.Name == "DateTimeOffset")
                        return "google.protobuf.Timestamp";
                    if (type.Name == "TimeSpan")
                        return "int64";
                    
                    // 對於複雜類型，簡化處理為字串
                    // 實際應用中，應為複雜類型生成對應的 message 定義
                    return "string";
            }
        }

        private string ToCamelCase(string input)
        {
            if (string.IsNullOrEmpty(input) || !char.IsUpper(input[0]))
                return input;
            
            return char.ToLower(input[0]) + input.Substring(1);
        }

        private string ToPascalCase(string input)
        {
            if (string.IsNullOrEmpty(input) || !char.IsLower(input[0]))
                return input;
            
            return char.ToUpper(input[0]) + input.Substring(1);
        }

        /// <summary>
        /// 語法接收器，用於收集標記了 GenerateGrpcServiceAttribute 的類
        /// </summary>
        private class SyntaxReceiver : ISyntaxContextReceiver
        {
            public List<(INamedTypeSymbol ClassSymbol, INamedTypeSymbol InterfaceSymbol)> CandidateClasses { get; } = 
                new List<(INamedTypeSymbol, INamedTypeSymbol)>();

            public void OnVisitSyntaxNode(GeneratorSyntaxContext context)
            {
                // 只處理類聲明
                if (context.Node is ClassDeclarationSyntax classDeclaration &&
                    classDeclaration.AttributeLists.Count > 0)
                {
                    SemanticModel semanticModel = context.SemanticModel;
                    INamedTypeSymbol classSymbol = semanticModel.GetDeclaredSymbol(classDeclaration) as INamedTypeSymbol;
                    
                    if (classSymbol == null)
                        return;
                    
                    // 檢查是否標記了目標 Attribute
                    foreach (AttributeData attribute in classSymbol.GetAttributes())
                    {
                        if (attribute.AttributeClass?.ToDisplayString() == AttributeFullName &&
                            attribute.ConstructorArguments.Length > 0)
                        {
                            if (attribute.ConstructorArguments[0].Value is INamedTypeSymbol interfaceSymbol)
                            {
                                CandidateClasses.Add((classSymbol, interfaceSymbol));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
} 