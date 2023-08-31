// See https://aka.ms/new-console-template for more information

using Grpc.Net.Client;
using ProtoBuf.Grpc.Client;
using Shared.Contracts;


async Task Sample1Client()
{
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
}

async Task Sample2Client()
{
    var channel = GrpcChannel.ForAddress("http://localhost:5000");
    var client1 = channel.CreateGrpcService<IMyAmazingService>();
    var greeting = await client1.SearchAsync(new SearchRequest
    {
        Name = "GreeterClient"
    });
    Console.WriteLine(greeting.Id);
    Console.WriteLine("Press any key to exit...");
    Console.ReadKey();    
}

await Sample2Client();