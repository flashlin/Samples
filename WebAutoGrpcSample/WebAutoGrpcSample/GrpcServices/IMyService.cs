using System.Runtime.Serialization;
using System.ServiceModel;
using Grpc.Core;
using Grpc.Net.Client;
using ProtoBuf.Grpc;
using ProtoBuf.Grpc.Client;
using ProtoBuf.Grpc.Configuration;
using ProtoBuf.Grpc.Server;

namespace WebAutoGrpcSample.GrpcServices;

//dotnet tool install -g dotnet-grpc
public interface IMyService
{
    public string SayHello(string message);
}

[ServiceContract]
[Service("foo")]
public interface IMyAmazingService {
    ValueTask<SearchResponse> SearchAsync(SearchRequest request);
}

[DataContract]
public class SearchRequest
{
    [DataMember(Order = 1)]
    public string Name { get; set; }
}

public class SearchResponse
{
    [DataMember(Order = 1)]
    public int Id { get; set; }
}

public class MyServer : IMyAmazingService {
    public ValueTask<SearchResponse> SearchAsync(SearchRequest request)
    {
        throw new NotImplementedException();
    }
}

public class Sample1
{
    public async Task Test()
    {
        GrpcClientFactory.AllowUnencryptedHttp2 = true;
        using var channel = GrpcChannel.ForAddress("http://localhost:10042");
        var calculator = channel.CreateGrpcService<IMyAmazingService>();
        var result = await calculator.SearchAsync(new SearchRequest
        {
            Name = null
        });
        Console.WriteLine(result.Id);
    }

    public async Task CreateService()
    {
        const int port = 10042;
        var server = new Server
        {
            Ports = { new ServerPort("localhost", port, ServerCredentials.Insecure) }
        };
        server.Services.AddCodeFirst<IMyAmazingService>(new MyServer());
        server.Start();

        Console.WriteLine("server listening on port " + port);
        Console.ReadKey();

        await server.ShutdownAsync();
    }
}