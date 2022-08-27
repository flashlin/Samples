using Serilog;
using Serilog.Formatting.Compact;
using Serilog.Sinks.Grafana.Loki;


Serilog.Debugging.SelfLog.Enable(Console.Error);

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((ctx, cfg) =>
{
	cfg.ReadFrom.Configuration(ctx.Configuration);
	cfg.Enrich.WithProperty("Application", ctx.HostingEnvironment.ApplicationName)
		.Enrich.WithProperty("Environment", ctx.HostingEnvironment.EnvironmentName);
		//.WriteTo.Console(new RenderedCompactJsonFormatter())
		//.WriteTo.GrafanaLoki(ctx.Configuration["loki"]);
});

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

app.UseSerilogRequestLogging();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
	app.UseSwagger();
	app.UseSwaggerUI();
}

app.UseAuthorization();

app.MapControllers();

app.Run();
