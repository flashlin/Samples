using ChatTrainDataWeb.Models.Repositories;
using Microsoft.OpenApi.Models;

var builder = WebApplication.CreateBuilder(args);
var services = builder.Services;
services.AddTransient<ChatTrainDataContext>();
services.AddScoped<IChatTrainDataRepo, ChatTrainDataRepo>();
services.AddControllersWithViews()
    .AddRazorRuntimeCompilation();
services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo {Title = "Create Train Char Data API", Version = "v1"});
});

var app = builder.Build();

//app.MapGet("/", () => "Hello World!");
app.UseSwagger();
app.UseSwaggerUI(c => { c.SwaggerEndpoint("/swagger/v1/swagger.json", "Your API v1"); });
app.MapControllers();

app.Run();