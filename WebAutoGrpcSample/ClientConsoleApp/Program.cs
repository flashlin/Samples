// See https://aka.ms/new-console-template for more information

using Grpc.Net.Client;
using ProtoBuf.Grpc.Client;
using Shared.Contracts;

using var channel = GrpcChannel.ForAddress("http://localhost:10042");
var client = channel.CreateGrpcService<IMyAmazingService>();

var reply = await client.SearchAsync(
    new SearchRequest
    {
        Name = "GreeterClient"
    });

Console.WriteLine($"Greeting: {reply.Id}");
Console.WriteLine("Press any key to exit...");
Console.ReadKey();