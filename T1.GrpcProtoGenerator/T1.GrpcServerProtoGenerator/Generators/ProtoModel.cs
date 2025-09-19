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
            => Messages.FirstOrDefault(m => m.Name == name) ?? 
               Messages.FirstOrDefault(m => m.GetFullName() == name);

        public ProtoEnum FindEnum(string name)
        {
            return Enums.FirstOrDefault(m=>m.Name == name);
        }

        public string FindRpcFullTypename(string name)
        {
            var msg = Messages.FirstOrDefault(m => m.Name == name);
            if (msg != null)
            {
                return msg.GetFullName();
            }
            var protoEnum = Enums.FirstOrDefault(m => m.Name == name);
            if (protoEnum != null)
            {
                return protoEnum.GetFullName();
            }
            return name;
        }
        
        public string FindCsharpTypeName(string protoTypeName)
        {
            var message = FindMessage(protoTypeName);
            if (message != null)
            {
                return message.GetCsharpTypeName();
            }
            
            var protoEnum = FindEnum(protoTypeName);
            if (protoEnum != null)
            {
                return protoEnum.GetCsharpTypeName();
            }

            return null;
        }

        public string FindCsharpTypeFullName(string protoTypeName)
        {
            var message = FindMessage(protoTypeName);
            if (message != null)
            {
                return message.GetCsharpTypeFullname();
            }
            
            var protoEnum = FindEnum(protoTypeName);
            if (protoEnum != null)
            {
                return protoEnum.GetCsharpTypeFullname();
            }

            return null;
        }
    }

    internal class ProtoMessage
    {
        public string Name { get; set; } = string.Empty;

        public string GetCsharpTypeFullname()
        {
            return $"{CsharpNamespace}.{GetCsharpTypeName()}";
        }
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<ProtoField> Fields { get; } = new List<ProtoField>();
        public string ProtoPath { get; set; } = string.Empty;

        public string GetFullName()
        {
            return $"{CsharpNamespace}.{Name}"; 
        }

        public string GetCsharpTypeName()
        {
            return $"{Name}GrpcDto";
        }
    }

    internal class ProtoField
    {
        public int Tag { get; set; }
        public string Type { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public bool IsRepeated { get; set; } = false;
        public bool IsOption { get; set; } = false;
    }

    internal class ProtoService
    {
        public string Name { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<ProtoRpc> Rpcs { get; } = new List<ProtoRpc>();
        public string ProtoPath { get; set; } = string.Empty;
    }

    internal class ProtoRpc
    {
        public string Name { get; set; } = string.Empty;
        public string RequestType { get; set; } = string.Empty;
        public string RequestFullTypename { get; set; } = string.Empty;
        public string ResponseType { get; set; } = string.Empty;
        public string ResponseFullTypename { get; set; } = string.Empty;
    }

    internal class ProtoEnum
    {
        public string Name { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<(string Name, int Value)> Values { get; } = new List<(string, int)>();
        public string ProtoPath { get; set; } = string.Empty;

        public string GetCsharpTypeFullname()
        {
            return $"{CsharpNamespace}.{GetCsharpTypeName()}";
        }

        public string GetCsharpTypeName()
        {
            return $"{Name}GrpcEnum";
        }

        public string GetFullName()
        {
            return $"{CsharpNamespace}.{Name}";
        }
    }
}
