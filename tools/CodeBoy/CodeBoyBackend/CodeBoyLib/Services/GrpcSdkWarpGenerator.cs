using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace CodeBoyLib.Services
{
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

