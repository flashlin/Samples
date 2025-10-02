using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using T1.Standard.IO;

namespace CodeBoyLib.Services
{
    public class PropertyInfo
    {
        public string CsharpTypeName { get; set; } = string.Empty;
        public string PropertyName { get; set; } = string.Empty;
    }

    public class MethodInfo
    {
        public string MethodName { get; set; } = string.Empty;
        public List<PropertyInfo> InputType { get; set; } = new List<PropertyInfo>();
        public List<PropertyInfo> ResponseType { get; set; } = new List<PropertyInfo>();
    }

    public class GrpcSdkWarpGenerator
    {
        public List<Type> QueryGrpcClientTypes(string assemblyFile)
        {
            var assembly = Assembly.LoadFrom(assemblyFile);
            var clientTypes = new List<Type>();

            foreach (var type in assembly.GetTypes())
            {
                if (IsGrpcClientType(type))
                {
                    clientTypes.Add(type);
                }
            }

            return clientTypes;
        }

        public List<MethodInfo> QueryWarpInterfaceFromGrpcClientType(Type grpcClientType)
        {
            var result = new List<MethodInfo>();
            var methods = grpcClientType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(m => m.IsVirtual && !m.IsFinal);

            foreach (var method in methods)
            {
                var methodInfo = new MethodInfo
                {
                    MethodName = method.Name
                };

                var parameters = method.GetParameters();
                if (parameters.Length > 0)
                {
                    var requestParam = parameters[0];
                    methodInfo.InputType = ExtractProperties(requestParam.ParameterType);
                }

                var returnType = method.ReturnType;
                if (returnType.IsGenericType)
                {
                    var genericArgs = returnType.GetGenericArguments();
                    if (genericArgs.Length > 0)
                    {
                        methodInfo.ResponseType = ExtractProperties(genericArgs[0]);
                    }
                }
                else if (returnType != typeof(void))
                {
                    methodInfo.ResponseType = ExtractProperties(returnType);
                }

                result.Add(methodInfo);
            }

            return result;
        }

        public string GenProxyInterfaceCode(Type grpcClientType)
        {
            var grpcClientMembers = QueryWarpInterfaceFromGrpcClientType(grpcClientType);
            var namespaceName = grpcClientType.Namespace + "Warp";
            var typeName = grpcClientType.Name;
            var output = new IndentStringBuilder();

            WriteNamespaceStart(output, namespaceName);
            WriteProxyInterface(output, typeName, grpcClientMembers);
            WriteProxyClass(output, typeName, grpcClientMembers);
            WriteNamespaceEnd(output);

            return output.ToString();
        }

        private void WriteNamespaceStart(IndentStringBuilder output, string namespaceName)
        {
            output.WriteLine($"namespace {namespaceName}");
            output.WriteLine("{");
            output.Indent++;
        }

        private void WriteNamespaceEnd(IndentStringBuilder output)
        {
            output.Indent--;
            output.WriteLine("}");
        }

        private void WriteProxyInterface(IndentStringBuilder output, string typeName, List<MethodInfo> members)
        {
            output.WriteLine($"public interface I{typeName}Proxy");
            output.WriteLine("{");
            output.Indent++;

            foreach (var member in members)
            {
                WriteInterfaceMethod(output, member);
            }

            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void WriteInterfaceMethod(IndentStringBuilder output, MethodInfo member)
        {
            var returnTypeName = GetReturnTypeName(member);
            var inputTypeName = GetInputTypeName(member);

            if (!string.IsNullOrEmpty(inputTypeName))
            {
                output.WriteLine($"{returnTypeName} {member.MethodName}({inputTypeName} req);");
            }
            else
            {
                output.WriteLine($"{returnTypeName} {member.MethodName}();");
            }
        }

        private void WriteProxyClass(IndentStringBuilder output, string typeName, List<MethodInfo> members)
        {
            output.WriteLine($"public class {typeName}Proxy : I{typeName}Proxy");
            output.WriteLine("{");
            output.Indent++;

            foreach (var member in members)
            {
                WriteClassMethod(output, member);
            }

            output.Indent--;
            output.WriteLine("}");
        }

        private void WriteClassMethod(IndentStringBuilder output, MethodInfo member)
        {
            var returnTypeName = GetReturnTypeName(member);
            var inputTypeName = GetInputTypeName(member);

            WriteMethodSignature(output, returnTypeName, member.MethodName, inputTypeName);
            WriteMethodBody(output);
        }

        private void WriteMethodSignature(IndentStringBuilder output, string returnTypeName, string methodName, string inputTypeName)
        {
            if (!string.IsNullOrEmpty(inputTypeName))
            {
                output.WriteLine($"public {returnTypeName} {methodName}({inputTypeName} req)");
            }
            else
            {
                output.WriteLine($"public {returnTypeName} {methodName}()");
            }
        }

        private void WriteMethodBody(IndentStringBuilder output)
        {
            output.WriteLine("{");
            output.Indent++;
            output.WriteLine("throw new NotImplementedException();");
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private string GetReturnTypeName(MethodInfo member)
        {
            return member.ResponseType.Count > 0 
                ? BuildTypeName(member.ResponseType) 
                : "Task";
        }

        private string GetInputTypeName(MethodInfo member)
        {
            return member.InputType.Count > 0 
                ? BuildTypeName(member.InputType) 
                : string.Empty;
        }

        private string BuildTypeName(List<PropertyInfo> properties)
        {
            if (properties.Count == 0)
                return string.Empty;

            if (properties.Count == 1)
                return properties[0].CsharpTypeName;

            return "object";
        }

        private List<PropertyInfo> ExtractProperties(Type type)
        {
            var properties = new List<PropertyInfo>();
            var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);

            foreach (var prop in props)
            {
                properties.Add(new PropertyInfo
                {
                    CsharpTypeName = GetCsharpTypeName(prop.PropertyType),
                    PropertyName = prop.Name
                });
            }

            return properties;
        }

        private string GetCsharpTypeName(Type type)
        {
            if (type == typeof(int)) return "int";
            if (type == typeof(long)) return "long";
            if (type == typeof(short)) return "short";
            if (type == typeof(byte)) return "byte";
            if (type == typeof(bool)) return "bool";
            if (type == typeof(float)) return "float";
            if (type == typeof(double)) return "double";
            if (type == typeof(decimal)) return "decimal";
            if (type == typeof(string)) return "string";
            if (type == typeof(void)) return "void";

            if (type.IsGenericType)
            {
                var genericTypeDef = type.GetGenericTypeDefinition();
                var genericArgs = type.GetGenericArguments();
                var genericArgNames = string.Join(", ", genericArgs.Select(GetCsharpTypeName));
                var typeName = genericTypeDef.Name;
                var backtickIndex = typeName.IndexOf('`');
                if (backtickIndex > 0)
                {
                    typeName = typeName.Substring(0, backtickIndex);
                }
                return $"{typeName}<{genericArgNames}>";
            }

            if (type.IsArray)
            {
                var elementType = type.GetElementType();
                return $"{GetCsharpTypeName(elementType)}[]";
            }

            return type.Name;
        }

        private bool IsGrpcClientType(Type type)
        {
            if (type == null || type.BaseType == null)
                return false;

            var baseType = type.BaseType;
            
            if (!baseType.IsGenericType)
                return false;

            var genericTypeDef = baseType.GetGenericTypeDefinition();
            if (genericTypeDef.Name != "ClientBase`1")
                return false;

            var genericArgs = baseType.GetGenericArguments();
            return genericArgs.Length == 1 && genericArgs[0] == type;
        }
    }
}

