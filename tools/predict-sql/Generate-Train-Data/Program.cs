using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
// See https://aka.ms/new-console-template for more information

// ConsoleHost builder
var services = new ServiceCollection();

// Register services
services.AddLogging(configure => configure.AddConsole());
services.AddSingleton<IGenerateTrainDataService, GenerateTrainDataService>();

var serviceProvider = services.BuildServiceProvider();

// Example: resolve and use the service
var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
var trainDataService = serviceProvider.GetRequiredService<IGenerateTrainDataService>();
trainDataService.Run();


logger.LogInformation("ConsoleHost started.");
Console.WriteLine("Hello World!");