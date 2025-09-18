// ProtoModel.cs

using System.Collections.Generic;
using System.Linq;

namespace T1.GrpcProtoGenerator.Generators
{
    internal class ProtoModel
    {
        public List<string> Imports { get; } = new List<string>();
        public List<ProtoMessage> Messages { get; } = new List<ProtoMessage>();
        public List<ProtoService> Services { get; } = new List<ProtoService>();
        public List<ProtoEnum> Enums { get; } = new List<ProtoEnum>();

        public ProtoMessage FindMessage(string name)
            => Messages.FirstOrDefault(m => m.Name == name) ?? Messages.FirstOrDefault(m => m.FullName == name);

        public string GetTatgetNamespace()
        {
            // Get namespace from first available element
            var firstMessage = Messages.FirstOrDefault();
            if (firstMessage != null && !string.IsNullOrEmpty(firstMessage.CsharpNamespace))
                return firstMessage.CsharpNamespace;
                
            var firstEnum = Enums.FirstOrDefault();
            if (firstEnum != null && !string.IsNullOrEmpty(firstEnum.CsharpNamespace))
                return firstEnum.CsharpNamespace;
                
            var firstService = Services.FirstOrDefault();
            if (firstService != null && !string.IsNullOrEmpty(firstService.CsharpNamespace))
                return firstService.CsharpNamespace;
                
            return "Generated";
        }
    }

    internal class ProtoMessage
    {
        public string Name { get; set; } = string.Empty;
        public string FullName { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<ProtoField> Fields { get; } = new List<ProtoField>();
    }

    internal class ProtoField
    {
        public int Tag { get; set; }
        public string Type { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public bool IsRepeated { get; set; } = false;
    }

    internal class ProtoService
    {
        public string Name { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<ProtoRpc> Rpcs { get; } = new List<ProtoRpc>();
    }

    internal class ProtoRpc
    {
        public string Name { get; set; } = string.Empty;
        public string RequestType { get; set; } = string.Empty;
        public string ResponseType { get; set; } = string.Empty;
    }

    internal class ProtoEnum
    {
        public string Name { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<(string Name, int Value)> Values { get; } = new List<(string, int)>();
    }
}
