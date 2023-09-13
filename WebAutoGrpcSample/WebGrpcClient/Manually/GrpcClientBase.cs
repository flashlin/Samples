namespace WebGrpcClient.Manually;
using grpc = global::Grpc.Core;

public partial class GrpcClientBase : grpc::ClientBase<GrpcClientBase>
{
    static readonly string __ServiceName = "BettingBudget";
    
    [global::System.CodeDom.Compiler.GeneratedCode("grpc_csharp_plugin", null)]
    public GrpcClientBase(grpc::ChannelBase channel) : base(channel)
    {
    }
    [global::System.CodeDom.Compiler.GeneratedCode("grpc_csharp_plugin", null)]
    public GrpcClientBase(grpc::CallInvoker callInvoker) : base(callInvoker)
    {
    }
    [global::System.CodeDom.Compiler.GeneratedCode("grpc_csharp_plugin", null)]
    protected GrpcClientBase() : base()
    {
    }
    [global::System.CodeDom.Compiler.GeneratedCode("grpc_csharp_plugin", null)]
    protected GrpcClientBase(ClientBaseConfiguration configuration) : base(configuration)
    {
    }
    protected override GrpcClientBase NewInstance(ClientBaseConfiguration configuration)
    {
        return new GrpcClientBase(configuration);
    }
    
    
    static void __Helper_SerializeMessage(global::Google.Protobuf.IMessage message, grpc::SerializationContext context)
    {
#if !GRPC_DISABLE_PROTOBUF_BUFFER_SERIALIZATION
        if (message is global::Google.Protobuf.IBufferMessage)
        {
            context.SetPayloadLength(message.CalculateSize());
            global::Google.Protobuf.MessageExtensions.WriteTo(message, context.GetBufferWriter());
            context.Complete();
            return;
        }
#endif
        context.Complete(global::Google.Protobuf.MessageExtensions.ToByteArray(message));
    }
    
    static class __Helper_MessageCache<T>
    {
        public static readonly bool IsBufferMessage = global::System.Reflection.IntrospectionExtensions.GetTypeInfo(typeof(global::Google.Protobuf.IBufferMessage)).IsAssignableFrom(typeof(T));
    }

    static T __Helper_DeserializeMessage<T>(grpc::DeserializationContext context, global::Google.Protobuf.MessageParser<T> parser) where T : global::Google.Protobuf.IMessage<T>
    {
#if !GRPC_DISABLE_PROTOBUF_BUFFER_SERIALIZATION
        if (__Helper_MessageCache<T>.IsBufferMessage)
        {
            return parser.ParseFrom(context.PayloadAsReadOnlySequence());
        }
#endif
        return parser.ParseFrom(context.PayloadAsNewBuffer());
    }

    static readonly grpc::Marshaller<global::WebGrpcContract.HelloRequest> __Marshaller_BettingBudgetRequest = 
        grpc::Marshallers.Create(__Helper_SerializeMessage, context => __Helper_DeserializeMessage(context, global::MemberCenterApi.Messages.BettingBudgetRequest.Parser));
    
    static readonly grpc::Method<global::WebGrpcContract.HelloRequest, global::WebGrpcContract.HelloResponse> __Method_Get = 
        new grpc::Method<global::WebGrpcContract.HelloRequest, global::WebGrpcContract.HelloResponse>(
        grpc::MethodType.Unary,
        __ServiceName,
        "Get",
        __Marshaller_BettingBudgetRequest,
        __Marshaller_BettingBudgetResponse);

    
    public virtual global::WebGrpcContract.HelloResponse SayHello(global::WebGrpcContract.HelloRequest request)
    {
        grpc::Metadata headers = null;
        global::System.DateTime? deadline = null;
        var cancellationToken = default(global::System.Threading.CancellationToken);
        var options = new grpc::CallOptions(headers, deadline, cancellationToken);
        return CallInvoker.BlockingUnaryCall(__Method_Get, null, options, request);
    }

}