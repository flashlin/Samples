using System.Net;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using ProtoBuf.Grpc.Server;
using WebAutoGrpcSample.GrpcServices;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options =>
{
    options.Listen(IPAddress.Any, 5000, listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http2;
    });
});
builder.Services.AddCodeFirstGrpc(config =>
{
    config.ResponseCompressionLevel = System.IO.Compression.CompressionLevel.Optimal;
});


var app = builder.Build();

app.MapGet("/", () => "Hello World!");

//var s = new Sample1();
//await s.CreateService();

app.MapGrpcService<MyServer>();

app.Run();