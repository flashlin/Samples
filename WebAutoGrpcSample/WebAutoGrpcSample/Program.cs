using WebAutoGrpcSample.GrpcServices;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "Hello World!");

var s = new Sample1();
await s.CreateService();

app.Run();