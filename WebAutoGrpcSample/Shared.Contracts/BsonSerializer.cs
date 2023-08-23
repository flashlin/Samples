using Bond;
using Bond.IO;
using Bond.IO.Safe;
using Bond.Protocols;
using Grpc.Core;

namespace Shared.Contracts;

public class BsonSerializer
{
    // public byte[] ToBytes<T>(T obj)
    // {
    //     var marshaller = new Marshaller<T>();
    //     var buffer = new OutputBuffer();
    //     var writer = new FastBinaryWriter<OutputBuffer>(buffer);
    //     Serialize.To(writer, obj);
    //     var output = new byte[buffer.Data.Count];
    //     Array.Copy(buffer.Data.Array!, 0, output, 0, (int)buffer.Position);
    //     return output;
    // }
    //
    // public T FromBytes<T>(byte[] bytes)
    // {
    //     var buffer = new InputBuffer(bytes);
    //     var data = Deserialize<T>.From(new FastBinaryReader<InputBuffer>(buffer));
    //     return data;
    // }
    //
    // static void Serialize<T>(T value, IOutputStream writer)
    // {
    //     var protocolWriter = new CompactBinaryWriter<OutputBuffer>(writer);
    //     var serializer = new Serializer<CompactBinaryWriter<OutputBuffer>>(typeof(T));
    //     serializer.Serialize(value, protocolWriter);
    // }
    //
    // static T Deserialize<T>(IInputStream reader)
    // {
    //     var protocolReader = new CompactBinaryReader<InputBuffer>(reader);
    //     var deserializer = new Deserializer<CompactBinaryReader<InputBuffer>>(typeof(T));
    //     return deserializer.Deserialize<T>(protocolReader);
    // }
    //
    //
    // public Marshaller<T> MarshallerFor<T>()
    // {
    //     byte[] Serializer(T clz) => Serialize(clz);
    //     T Deserializer(byte[] bytes) => Deserialize<T>(bytes);
    //     return new Marshaller<T>(Serializer, Deserializer);
    // }
    //
    // public byte[] Serialize<T>(T clz)
    // {
    //     var json = JsonSerializer.Serialize(clz);
    //     return System.Text.Encoding.UTF8.GetBytes(json);
    // }
    //
    // public T Deserialize<T>(byte[] bytes)
    // {
    //     using MemoryStream memoryStream = new MemoryStream(bytes);
    //     return JsonSerializer.Deserialize<T>(memoryStream)!;
    // }
    //
    //
    // private const string SERVICE_NAME = "YourServiceName"; // Replace with the actual service name
    //
    // private static Method<CreateRequest, CreateResponse> CREATE_METHOD =
    //     new Method<CreateRequest, CreateResponse>(
    //         type: MethodType.Unary,
    //         serviceName: SERVICE_NAME,
    //         name: "Create",
    //         requestMarshaller: Marshallers.Create(
    //             serializer: (request, stream) => 
    //                 ProtoBuf.Serializer.Serialize(stream, request),
    //             deserializer: stream => 
    //                 ProtoBuf.Serializer.Deserialize<CreateRequest>(stream)),
    //         responseMarshaller: Marshallers.Create(
    //             serializer: (response, stream) => 
    //                 ProtoBuf.Serializer.Serialize(stream, response),
    //             deserializer: stream => 
    //                 ProtoBuf.Serializer.Deserialize<CreateResponse>(stream)));
    //
    // public async Task<CreateResponse> Create(CreateRequest request, Channel channel)
    // {
    //     var callInvoker = channel.CreateCallInvoker();
    //
    //     var call = callInvoker.AsyncUnaryCall(
    //         CREATE_METHOD,
    //         null,
    //         new CallOptions(deadline: DateTime.UtcNow.AddSeconds(30)),
    //         request);
    //
    //     return await call.ResponseAsync;
    // }
}