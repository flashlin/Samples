// ProtoModel.cs

using System.Collections.Generic;
using System.Linq;

namespace T1.GrpcProtoGenerator.Generators
{
    public class ProtoModel
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
            
            // Handle built-in protobuf types
            if (IsBuiltInProtobufType(name))
            {
                return GetBuiltInTypeFullname(name);
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

            // Handle built-in protobuf types
            if (IsBuiltInProtobufType(protoTypeName))
            {
                return GetBuiltInTypeCsharpName(protoTypeName);
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

        private static bool IsBuiltInProtobufType(string typeName)
        {
            return typeName.StartsWith("google.protobuf.");
        }

        private static string GetBuiltInTypeFullname(string typeName)
        {
            // For google.protobuf types, return the fully qualified .NET type name
            return typeName switch
            {
                "google.protobuf.Empty" => "Google.Protobuf.WellKnownTypes.Empty",
                "google.protobuf.Timestamp" => "Google.Protobuf.WellKnownTypes.Timestamp",
                "google.protobuf.Duration" => "Google.Protobuf.WellKnownTypes.Duration",
                "google.protobuf.Any" => "Google.Protobuf.WellKnownTypes.Any",
                "google.protobuf.Value" => "Google.Protobuf.WellKnownTypes.Value",
                "google.protobuf.Struct" => "Google.Protobuf.WellKnownTypes.Struct",
                "google.protobuf.ListValue" => "Google.Protobuf.WellKnownTypes.ListValue",
                "google.protobuf.NullValue" => "Google.Protobuf.WellKnownTypes.NullValue",
                "google.protobuf.StringValue" => "Google.Protobuf.WellKnownTypes.StringValue",
                "google.protobuf.Int32Value" => "Google.Protobuf.WellKnownTypes.Int32Value",
                "google.protobuf.Int64Value" => "Google.Protobuf.WellKnownTypes.Int64Value",
                "google.protobuf.UInt32Value" => "Google.Protobuf.WellKnownTypes.UInt32Value",
                "google.protobuf.UInt64Value" => "Google.Protobuf.WellKnownTypes.UInt64Value",
                "google.protobuf.BoolValue" => "Google.Protobuf.WellKnownTypes.BoolValue",
                "google.protobuf.BytesValue" => "Google.Protobuf.WellKnownTypes.BytesValue",
                "google.protobuf.DoubleValue" => "Google.Protobuf.WellKnownTypes.DoubleValue",
                "google.protobuf.FloatValue" => "Google.Protobuf.WellKnownTypes.FloatValue",
                _ => typeName
            };
        }

        private static string GetBuiltInTypeCsharpName(string typeName)
        {
            // For DTO types, we don't want to use the built-in types directly
            // Instead, we'll create corresponding DTO types or use special handling
            return typeName switch
            {
                "google.protobuf.Empty" => "EmptyGrpcDto", // Special case for Empty
                "google.protobuf.Timestamp" => "DateTime", // Map to DateTime for DTOs
                "google.protobuf.Duration" => "TimeSpan", // Map to TimeSpan for DTOs
                _ => $"{typeName.Replace("google.protobuf.", "")}GrpcDto"
            };
        }
    }

    public class ProtoMessage
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

    public class ProtoField
    {
        public int Tag { get; set; }
        public string Type { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public bool IsRepeated { get; set; } = false;
        public bool IsOption { get; set; } = false;
    }

    public class ProtoService
    {
        public string Name { get; set; } = string.Empty;
        public string CsharpNamespace { get; set; } = string.Empty;
        public List<ProtoRpc> Rpcs { get; } = new List<ProtoRpc>();
        public string ProtoPath { get; set; } = string.Empty;
    }

    public class ProtoRpc
    {
        public string Name { get; set; } = string.Empty;
        public string RequestType { get; set; } = string.Empty;
        public string RequestFullTypename { get; set; } = string.Empty;
        public string ResponseType { get; set; } = string.Empty;
        public string ResponseFullTypename { get; set; } = string.Empty;
    }

    public class ProtoEnum
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
