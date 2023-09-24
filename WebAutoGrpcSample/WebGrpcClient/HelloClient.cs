using System.Threading.Channels;
using Grpc.Core;
using WebGrpcContract;
using Google.Protobuf;
using Google.Protobuf.Reflection;


public class HelloClient
{
    public async Task Run()
    {
        var channel = new Grpc.Core.Channel("localhost:50051", ChannelCredentials.Insecure);
        CallInvoker invoker = new DefaultCallInvoker(channel);

        // 建立一個新的 Method
        Method<NewHelloRequest, NewHelloRequest> method = new Method<NewHelloRequest, NewHelloRequest>(
            MethodType.Unary,
            "helloworld.Greeter",
            "SayHello",
            MessageParser<NewHelloRequest>,
            NewHelloRequestParser.Default);

        // 建立一個新的 HelloRequest
        HelloRequest request = new HelloRequest { Name = "World" };

        // 呼叫 SayHello 方法並取得回應
        NewHelloRequest reply = await invoker.AsyncUnaryCall(method, null, new CallOptions(), request);

        Console.WriteLine("Greeting: " + reply.Message);

        // 關閉 Channel
        await channel.ShutdownAsync();
    }
}

public class NewHelloRequest : IMessage<NewHelloRequest>
{
    public string Name { get; set; }

    public NewHelloRequest()
    {
    }

    public void MergeFrom(NewHelloRequest message)
    {
        if (message == null)
            return;

        Name = message.Name;
    }

    public void MergeFrom(CodedInputStream input)
    {
        uint tag;
        while ((tag = input.ReadTag()) != 0)
        {
            switch (tag)
            {
                case 10: // Field 1, Name
                    Name = input.ReadString();
                    break;
                default:
                    input.SkipLastField();
                    break;
            }
        }
    }

    public void WriteTo(CodedOutputStream output)
    {
        if (Name != null)
        {
            output.WriteRawTag(10);
            output.WriteString(Name);
        }
    }

    public int CalculateSize()
    {
        throw new NotImplementedException();
    }

    public MessageDescriptor Descriptor { get; }
    
    public bool Equals(NewHelloRequest? other)
    {
        if (other == null)
            return false;

        return Name == other.Name;
    }

    public NewHelloRequest Clone()
    {
        var clone = new NewHelloRequest();
        clone.MergeFrom(this);
        return clone;
    }
}

public class NewHelloRequestParser : MessageParser<NewHelloRequest>
{
    public static NewHelloRequestParser Default { get; } = new NewHelloRequestParser();

    // public override NewHelloRequest ParseFrom(byte[] data)
    // {
    //     return ParseFrom(new CodedInputStream(data));
    // }
    //
    // public override NewHelloRequest ParseFrom(CodedInputStream input)
    // {
    //     HelloRequest message = new HelloRequest();
    //     uint tag;
    //     while ((tag = input.ReadTag()) != 0)
    //     {
    //         switch (tag)
    //         {
    //             case 10: // Field 1, Name
    //                 message.Name = input.ReadString();
    //                 break;
    //             default:
    //                 input.SkipLastField();
    //                 break;
    //         }
    //     }
    //     return message;
    // }
}