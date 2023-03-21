using Grpc.Core;
using Grpc.Net.Client;
using ProtoBuf.Grpc.Client;
using ProtoBuf.Grpc.Server;

namespace WebAutoGrpcSample.GrpcServices;

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