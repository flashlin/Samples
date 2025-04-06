using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using T1.GrpcSourceGenerator;

namespace GenGrpcService
{
    /// <summary>
    /// gRPC 服務代碼生成器
    /// </summary>
    public class GrpcServiceSourceGenerator
    {
        private const string AttributeName = "GenerateGrpcServiceAttribute";
        private const string AttributeFullName = "T1.GrpcSourceGenerator.GenerateGrpcServiceAttribute";

        /// <summary>
        /// 執行 gRPC 服務代碼生成
        /// </summary>
        /// <param name="csprojFile">專案文件路徑</param>
        public void Execute(string csprojFile)
        {
            if (string.IsNullOrEmpty(csprojFile))
            {
                throw new ArgumentNullException(nameof(csprojFile));
            }

            Console.WriteLine("====================================");
            Console.WriteLine($"開始處理專案文件: {csprojFile}");
            Console.WriteLine("====================================");
            
            if (!File.Exists(csprojFile))
            {
                Console.WriteLine($"錯誤：找不到專案文件 {csprojFile}");
                return;
            }

            string? projectDirectory = Path.GetDirectoryName(csprojFile);
            if (string.IsNullOrEmpty(projectDirectory))
            {
                throw new InvalidOperationException("無法獲取專案目錄");
            }

            Console.WriteLine($"專案目錄: {projectDirectory}");
            
            try 
            {
                // 獲取所有 C# 檔案，直接通過文件系統而不使用 MSBuild
                var csharpFiles = GetCSharpFiles(projectDirectory);
                if (csharpFiles.Count == 0)
                {
                    Console.WriteLine("找不到 C# 檔案");
                    return;
                }

                Console.WriteLine($"已找到 {csharpFiles.Count} 個 C# 檔案:");
                foreach (var file in csharpFiles)
                {
                    Console.WriteLine($"  - {file}");
                }

                // 創建代碼分析工作區
                var compilation = CreateCompilation(csharpFiles);
                if (compilation == null)
                {
                    Console.WriteLine("無法創建代碼編譯器");
                    return;
                }
                Console.WriteLine("成功創建代碼編譯器");

                // 掃描標記了 [GenerateGrpcService] 特性的類
                var candidateClasses = FindCandidateClasses(compilation);
                if (candidateClasses.Count == 0)
                {
                    Console.WriteLine("找不到標記了 [GenerateGrpcService] 的類");
                    Console.WriteLine("請確保已添加 GenerateGrpcServiceAttribute 特性並正確引用");
                    return;
                }

                Console.WriteLine($"已找到 {candidateClasses.Count} 個標記的類");

                // 處理每個標記的類
                foreach (var (classSymbol, interfaceSymbol) in candidateClasses)
                {
                    ProcessClass(classSymbol, interfaceSymbol, projectDirectory);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"執行過程中發生錯誤: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        private List<string> GetCSharpFiles(string projectDirectory)
        {
            var result = new List<string>();

            // 遞迴搜索目錄下所有的 .cs 文件，排除 bin 和 obj 目錄
            foreach (var file in Directory.GetFiles(projectDirectory, "*.cs", SearchOption.AllDirectories))
            {
                if (!file.Contains("\\bin\\") && !file.Contains("\\obj\\"))
                {
                    result.Add(file);
                }
            }

            return result;
        }

        private Compilation CreateCompilation(List<string> sourceFiles)
        {
            // 創建語法樹
            var syntaxTrees = new List<SyntaxTree>();
            foreach (var file in sourceFiles)
            {
                try
                {
                    string code = File.ReadAllText(file);
                    var syntaxTree = CSharpSyntaxTree.ParseText(code);
                    syntaxTrees.Add(syntaxTree);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"解析文件 {file} 時發生錯誤：{ex.Message}");
                }
            }

            // 獲取基本引用
            var references = new List<MetadataReference>
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Task).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(GenerateGrpcServiceAttribute).Assembly.Location)
            };

            // 獲取 .NET Runtime 路徑
            string runtimePath = Path.GetDirectoryName(typeof(object).Assembly.Location);
            string[] assemblies = { 
                "System.Runtime.dll",
                "System.Collections.dll",
                "System.Text.RegularExpressions.dll",
                "System.Console.dll",
                "System.IO.dll"
            };

            foreach (var assembly in assemblies)
            {
                string path = Path.Combine(runtimePath, assembly);
                if (File.Exists(path))
                {
                    references.Add(MetadataReference.CreateFromFile(path));
                }
            }

            // 創建編譯
            var compilationOptions = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary);
            var compilation = CSharpCompilation.Create(
                "DynamicAssembly",
                syntaxTrees: syntaxTrees,
                references: references,
                options: compilationOptions);

            return compilation;
        }

        private List<(INamedTypeSymbol ClassSymbol, INamedTypeSymbol InterfaceSymbol)> FindCandidateClasses(Compilation compilation)
        {
            var result = new List<(INamedTypeSymbol ClassSymbol, INamedTypeSymbol InterfaceSymbol)>();

            foreach (SyntaxTree syntaxTree in compilation.SyntaxTrees)
            {
                SemanticModel semanticModel = compilation.GetSemanticModel(syntaxTree);
                
                // 查找類聲明
                var classDeclarations = syntaxTree.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>();
                
                foreach (var classDeclaration in classDeclarations)
                {
                    var classSymbol = semanticModel.GetDeclaredSymbol(classDeclaration) as INamedTypeSymbol;
                    if (classSymbol == null)
                        continue;

                    // 尋找具有目標特性的類
                    foreach (var attribute in classSymbol.GetAttributes())
                    {
                        Console.WriteLine($"檢查類 {classSymbol.Name} 的特性: {attribute.AttributeClass?.Name}");
                        
                        if (attribute.AttributeClass?.Name == AttributeName || 
                            attribute.AttributeClass?.ToDisplayString() == AttributeFullName)
                        {
                            if (attribute.ConstructorArguments.Length > 0)
                            {
                                var attributeArg = attribute.ConstructorArguments[0].Value;
                                Console.WriteLine($"  特性參數類型: {attributeArg?.GetType().Name}");
                                
                                if (attributeArg is INamedTypeSymbol interfaceType)
                                {
                                    result.Add((classSymbol, interfaceType));
                                    Console.WriteLine($"找到類 {classSymbol.Name}，標記了接口 {interfaceType.Name}");
                                }
                                else
                                {
                                    Console.WriteLine($"  特性參數不是 INamedTypeSymbol");
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        private void ProcessClass(INamedTypeSymbol classSymbol, INamedTypeSymbol interfaceSymbol, string projectDirectory)
        {
            string className = classSymbol.Name;
            string interfaceName = interfaceSymbol.Name;
            
            // 從接口名稱獲取服務名稱 (去掉開頭的 'I')
            string serviceName = interfaceName.StartsWith("I") ? interfaceName.Substring(1) : interfaceName;
            
            Console.WriteLine($"處理 {className} 類，生成 {serviceName} 服務...");

            // 創建 GrpcProto 目錄 (如果不存在)
            string grpcProtoDir = Path.Combine(projectDirectory, "GrpcProto");
            if (!Directory.Exists(grpcProtoDir))
            {
                Directory.CreateDirectory(grpcProtoDir);
                Console.WriteLine($"已創建 GrpcProto 目錄");
            }

            // 生成 .proto 文件
            string protoContent = GenerateProtoFile(interfaceSymbol, serviceName);
            string protoFilePath = Path.Combine(grpcProtoDir, $"{serviceName}Messages.proto");
            File.WriteAllText(protoFilePath, protoContent);
            Console.WriteLine($"已生成 .proto 文件：{protoFilePath}");

            // 創建 GrpcServices 目錄 (如果不存在)
            string grpcServicesDir = Path.Combine(projectDirectory, "GrpcServices");
            if (!Directory.Exists(grpcServicesDir))
            {
                Directory.CreateDirectory(grpcServicesDir);
                Console.WriteLine($"已創建 GrpcServices 目錄");
            }

            // 生成 gRPC 服務實現類
            string serviceContent = GenerateGrpcServiceClass(classSymbol, interfaceSymbol, serviceName);
            string serviceFilePath = Path.Combine(grpcServicesDir, $"{serviceName}GrpcService.cs");
            File.WriteAllText(serviceFilePath, serviceContent);
            Console.WriteLine($"已生成 gRPC 服務實現類：{serviceFilePath}");
        }

        private string GenerateProtoFile(INamedTypeSymbol interfaceSymbol, string serviceName)
        {
            var protoBuilder = new StringBuilder();
            
            // Proto 文件頭
            protoBuilder.AppendLine("syntax = \"proto3\";");
            protoBuilder.AppendLine();
            
            // 命名空間/包名
            string packageName = interfaceSymbol.ContainingNamespace.ToDisplayString().ToLowerInvariant();
            protoBuilder.AppendLine($"package {packageName};");
            protoBuilder.AppendLine();
            
            // 導入常用的 proto 定義
            protoBuilder.AppendLine("import \"google/protobuf/timestamp.proto\";");
            protoBuilder.AppendLine("import \"google/protobuf/empty.proto\";");
            protoBuilder.AppendLine();
            
            // 跟踪已經生成的消息類型
            HashSet<string> generatedMessages = new HashSet<string>();
            
            // 為每個方法生成消息定義
            foreach (var member in interfaceSymbol.GetMembers())
            {
                if (member is IMethodSymbol methodSymbol && methodSymbol.MethodKind == MethodKind.Ordinary)
                {
                    GenerateMessagesForMethod(protoBuilder, methodSymbol, generatedMessages);
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

        private void GenerateMessagesForMethod(StringBuilder protoBuilder, IMethodSymbol methodSymbol, HashSet<string> generatedMessages)
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

        private string GenerateGrpcServiceClass(INamedTypeSymbol classSymbol, INamedTypeSymbol interfaceSymbol, string serviceName)
        {
            var serviceBuilder = new StringBuilder();
            
            // 添加必要的 using 語句
            serviceBuilder.AppendLine("using System;");
            serviceBuilder.AppendLine("using System.Threading.Tasks;");
            serviceBuilder.AppendLine("using Grpc.Core;");
            serviceBuilder.AppendLine($"using {interfaceSymbol.ContainingNamespace.ToDisplayString()};");
            serviceBuilder.AppendLine();
            
            // 命名空間
            string namespaceName = classSymbol.ContainingNamespace.ToDisplayString();
            serviceBuilder.AppendLine($"namespace {namespaceName}.GrpcServices");
            serviceBuilder.AppendLine("{");
            
            // 生成服務實現類
            serviceBuilder.AppendLine($"    /// <summary>");
            serviceBuilder.AppendLine($"    /// gRPC 服務實現，基於 {interfaceSymbol.Name}");
            serviceBuilder.AppendLine($"    /// </summary>");
            serviceBuilder.AppendLine($"    public class {serviceName}GrpcService : {serviceName}.{serviceName}Base");
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
            
            return serviceBuilder.ToString();
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
            if (string.IsNullOrEmpty(input))
                return string.Empty;
            
            if (!char.IsUpper(input[0]))
                return input;
            
            return char.ToLower(input[0]) + input[1..];
        }

        private string ToPascalCase(string input)
        {
            if (string.IsNullOrEmpty(input) || !char.IsLower(input[0]))
                return input;
            
            return char.ToUpper(input[0]) + input.Substring(1);
        }
    }
} 