using DemoServer.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddGrpc();
builder.Services.AddScoped<IGreeterGrpcService, GreeterService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.MapGrpcService<GreeterNativeGrpcService>();
app.MapGrpcService<Greeter2NativeGrpcService>();
app.MapGet("/",
    () =>
        "Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");

app.Run();

// Make Program class accessible for testing
public partial class Program { }

