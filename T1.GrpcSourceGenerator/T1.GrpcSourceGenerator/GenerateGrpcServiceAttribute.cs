using System;

namespace T1.GrpcSourceGenerator
{
    /// <summary>
    /// 標記需要生成 gRPC 服務的類別
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class GenerateGrpcServiceAttribute : Attribute
    {
        /// <summary>
        /// 獲取接口類型
        /// </summary>
        public Type InterfaceType { get; }

        /// <summary>
        /// 初始化 <see cref="GenerateGrpcServiceAttribute"/> 實例
        /// </summary>
        /// <param name="interfaceType">服務接口的類型</param>
        public GenerateGrpcServiceAttribute(Type interfaceType)
        {
            InterfaceType = interfaceType ?? throw new ArgumentNullException(nameof(interfaceType));
        }
    }
} 