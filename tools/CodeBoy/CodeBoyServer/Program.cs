using CodeBoyServer.ApiHandlers;
using CodeBoyServer.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() 
    { 
        Title = "CodeBoy Server API", 
        Version = "v1",
        Description = "API for generating Web API client code from Swagger specifications"
    });
});

// Register code generation service
builder.Services.AddScoped<ICodeGenService, GenWebApiClientService>();

// Add logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();

// Add CORS for development
builder.Services.AddCors(options =>
{
    options.AddPolicy("DevPolicy", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "CodeBoy Server API v1");
        c.RoutePrefix = string.Empty; // Set Swagger UI at the app's root
    });
    app.UseCors("DevPolicy");
}

app.UseHttpsRedirection();

// Configure API endpoints
CodeGenHandler.GenWebApiClient(app);

// Add a health check endpoint
app.MapGet("/health", () => Results.Ok(new { Status = "Healthy", Timestamp = DateTime.UtcNow }))
    .WithName("HealthCheck")
    .WithDescription("Health check endpoint")
    .WithTags("Health")
    .WithOpenApi();

app.Run();
