// ProtoParser.cs

using System.Text.RegularExpressions;

namespace T1.GrpcProtoGenerator.Generators
{
    internal static class ProtoParser
    {
        private static readonly Regex PackageRegex = new Regex(@"package\s+(?<pkg>[\w\.]+)\s*;", RegexOptions.Compiled);
        private static readonly Regex CsharpNamespaceRegex = new Regex(@"option\s+csharp_namespace\s*=\s*""(?<namespace>[^""]+)""\s*;", RegexOptions.Compiled);
        private static readonly Regex ImportRegex = new Regex(@"import\s+""(?<path>[^""]+)""\s*;", RegexOptions.Compiled);
        private static readonly Regex MessageRegex = new Regex(@"message\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex FieldRegex = new Regex(@"(?<repeated>repeated\s+)?(?<type>\w+)\s+(?<name>\w+)\s*=\s*(?<tag>\d+);", RegexOptions.Compiled);
        private static readonly Regex ServiceRegex = new Regex(@"service\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex RpcRegex = new Regex(@"rpc\s+(?<name>\w+)\s*\(\s*(?<req>\w+)\s*\)\s*returns\s*\(\s*(?<resp>\w+)\s*\)\s*;", RegexOptions.Compiled);
        private static readonly Regex EnumRegex = new Regex(@"enum\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex EnumFieldRegex = new Regex(@"(?<name>\w+)\s*=\s*(?<value>\d+);", RegexOptions.Compiled);

        public static ProtoModel ParseProtoText(string protoText)
        {
            var model = new ProtoModel();
            var packageName = ParsePackageAndNamespace(protoText, out var csharpNamespace);
            csharpNamespace = NormalCsharpNamespace(csharpNamespace);
            
            ParseImports(protoText, model);
            ParseMessages(protoText, model, packageName, csharpNamespace);
            ParseEnums(protoText, model, csharpNamespace);
            ParseServices(protoText, model, csharpNamespace);

            return model;
        }

        private static string ParsePackageAndNamespace(string protoText, out string csharpNamespace)
        {
            var packageMatch = PackageRegex.Match(protoText);
            var packageName = packageMatch.Success ? packageMatch.Groups["pkg"].Value : null;
            
            // Parse csharp_namespace option
            var csharpNamespaceMatch = CsharpNamespaceRegex.Match(protoText);
            csharpNamespace = csharpNamespaceMatch.Success ? csharpNamespaceMatch.Groups["namespace"].Value : packageName;

            return packageName;
        }

        private static void ParseImports(string protoText, ProtoModel model)
        {
            foreach (Match importMatch in ImportRegex.Matches(protoText))
            {
                var importPath = importMatch.Groups["path"].Value;
                model.Imports.Add(importPath);
            }
        }

        private static void ParseMessages(string protoText, ProtoModel model, string packageName, string csharpNamespace)
        {
            foreach (Match messageMatch in MessageRegex.Matches(protoText))
            {
                var name = messageMatch.Groups["name"].Value;
                var body = messageMatch.Groups["body"].Value;
                var protoMessage = new ProtoMessage 
                { 
                    Name = name, 
                    CsharpNamespace = csharpNamespace
                };
                
                ParseMessageFields(body, protoMessage);
                model.Messages.Add(protoMessage);
            }
        }

        private static string NormalCsharpNamespace(string protoText)
        {
            return protoText ?? "Generated"; 
        }

        private static void ParseMessageFields(string messageBody, ProtoMessage protoMessage)
        {
            foreach (Match fieldMatch in FieldRegex.Matches(messageBody))
            {
                protoMessage.Fields.Add(new ProtoField
                {
                    Type = fieldMatch.Groups["type"].Value,
                    Name = fieldMatch.Groups["name"].Value,
                    Tag = int.Parse(fieldMatch.Groups["tag"].Value),
                    IsRepeated = !string.IsNullOrEmpty(fieldMatch.Groups["repeated"].Value)
                });
            }
        }

        private static void ParseEnums(string protoText, ProtoModel model, string csharpNamespace)
        {
            foreach (Match enumMatch in EnumRegex.Matches(protoText))
            {
                var enumName = enumMatch.Groups["name"].Value;
                var body = enumMatch.Groups["body"].Value;
                var protoEnum = new ProtoEnum 
                { 
                    Name = enumName,
                    CsharpNamespace = csharpNamespace
                };
                
                ParseEnumFields(body, protoEnum);
                model.Enums.Add(protoEnum);
            }
        }

        private static void ParseEnumFields(string enumBody, ProtoEnum protoEnum)
        {
            foreach (Match enumFieldMatch in EnumFieldRegex.Matches(enumBody))
            {
                protoEnum.Values.Add((
                    enumFieldMatch.Groups["name"].Value, 
                    int.Parse(enumFieldMatch.Groups["value"].Value)
                ));
            }
        }

        private static void ParseServices(string protoText, ProtoModel model, string csharpNamespace)
        {
            foreach (Match serviceMatch in ServiceRegex.Matches(protoText))
            {
                var serviceName = serviceMatch.Groups["name"].Value;
                var serviceBody = serviceMatch.Groups["body"].Value;
                var protoService = new ProtoService 
                { 
                    Name = serviceName,
                    CsharpNamespace = csharpNamespace
                };
                
                ParseServiceRpcs(serviceBody, protoService);
                model.Services.Add(protoService);
            }
        }

        private static void ParseServiceRpcs(string serviceBody, ProtoService protoService)
        {
            foreach (Match rpcMatch in RpcRegex.Matches(serviceBody))
            {
                protoService.Rpcs.Add(new ProtoRpc
                {
                    Name = rpcMatch.Groups["name"].Value,
                    RequestType = rpcMatch.Groups["req"].Value,
                    ResponseType = rpcMatch.Groups["resp"].Value
                });
            }
        }
    }
}
