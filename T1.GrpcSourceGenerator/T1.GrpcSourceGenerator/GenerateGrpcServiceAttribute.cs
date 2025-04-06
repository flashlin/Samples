using System;

namespace T1.GrpcSourceGenerator
{
    /// <summary>
    /// 用於標記需要生成 gRPC 服務的類，並指定對應的接口類型
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class GenerateGrpcServiceAttribute : Attribute
    {
        /// <summary>
        /// 獲取服務接口類型
        /// </summary>
        public Type InterfaceType { get; }

        /// <summary>
        /// 初始化 <see cref="GenerateGrpcServiceAttribute"/> 的新實例
        /// </summary>
        /// <param name="interfaceType">gRPC 服務對應的接口類型</param>
        public GenerateGrpcServiceAttribute(Type interfaceType)
        {
            InterfaceType = interfaceType ?? throw new ArgumentNullException(nameof(interfaceType));
        }
    }
} 