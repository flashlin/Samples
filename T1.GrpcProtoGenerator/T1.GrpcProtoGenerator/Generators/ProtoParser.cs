// ProtoParser.cs

using System.Text.RegularExpressions;

namespace T1.GrpcProtoGenerator.Generators
{
    internal static class ProtoParser
    {
        private static readonly Regex PackageRegex = new Regex(@"package\s+(?<pkg>[\w\.]+)\s*;", RegexOptions.Compiled);
        private static readonly Regex MessageRegex = new Regex(@"message\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex FieldRegex = new Regex(@"(?<repeated>repeated\s+)?(?<type>\w+)\s+(?<name>\w+)\s*=\s*(?<tag>\d+);", RegexOptions.Compiled);
        private static readonly Regex ServiceRegex = new Regex(@"service\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex RpcRegex = new Regex(@"rpc\s+(?<name>\w+)\s*\(\s*(?<req>\w+)\s*\)\s*returns\s*\(\s*(?<resp>\w+)\s*\)\s*;", RegexOptions.Compiled);
        private static readonly Regex EnumRegex = new Regex(@"enum\s+(?<name>\w+)\s*\{(?<body>[\s\S]*?)\}", RegexOptions.Compiled);
        private static readonly Regex EnumFieldRegex = new Regex(@"(?<name>\w+)\s*=\s*(?<value>\d+);", RegexOptions.Compiled);

        public static ProtoModel ParseProtoText(string protoText)
        {
            var model = new ProtoModel();
            var packageMatch = PackageRegex.Match(protoText);
            string pkg = packageMatch.Success ? packageMatch.Groups["pkg"].Value : null;

            // Parse messages
            foreach (Match m in MessageRegex.Matches(protoText))
            {
                var name = m.Groups["name"].Value;
                var body = m.Groups["body"].Value;
                var pm = new ProtoMessage { Name = name, FullName = pkg != null ? pkg + "." + name : name };
                foreach (Match f in FieldRegex.Matches(body))
                {
                    pm.Fields.Add(new ProtoField
                    {
                        Type = f.Groups["type"].Value,
                        Name = f.Groups["name"].Value,
                        Tag = int.Parse(f.Groups["tag"].Value),
                        IsRepeated = !string.IsNullOrEmpty(f.Groups["repeated"].Value)
                    });
                }
                model.Messages.Add(pm);
            }

            // Parse enums
            foreach (Match e in EnumRegex.Matches(protoText))
            {
                var enumName = e.Groups["name"].Value;
                var body = e.Groups["body"].Value;
                var pe = new ProtoEnum { Name = enumName };
                foreach (Match ef in EnumFieldRegex.Matches(body))
                {
                    pe.Values.Add((ef.Groups["name"].Value, int.Parse(ef.Groups["value"].Value)));
                }
                model.Enums.Add(pe);
            }

            // Parse services
            foreach (Match s in ServiceRegex.Matches(protoText))
            {
                var sname = s.Groups["name"].Value;
                var sbody = s.Groups["body"].Value;
                var ps = new ProtoService { Name = sname };
                foreach (Match r in RpcRegex.Matches(sbody))
                {
                    ps.Rpcs.Add(new ProtoRpc
                    {
                        Name = r.Groups["name"].Value,
                        RequestType = r.Groups["req"].Value,
                        ResponseType = r.Groups["resp"].Value
                    });
                }
                model.Services.Add(ps);
            }

            return model;
        }
    }
}
