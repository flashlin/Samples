using CommandLine;
using MakeSwaggerSDK.Models;
using MakeSwaggerSDK.Services;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MakeSwaggerSDK
{
    // Command line options
    public class Options
    {
        [Value(0, MetaName = "swagger-url", Required = true, HelpText = "Swagger UI URL to parse")]
        public string SwaggerUrl { get; set; } = string.Empty;

        [Option('o', "output", Required = false, HelpText = "Output file name for generated SDK")]
        public string? OutputFileName { get; set; }

        [Option('n', "name", Required = false, HelpText = "SDK class name")]
        public string SdkName { get; set; } = "ApiClient";
    }

    class Program
    {
        static async Task<int> Main(string[] args)
        {
            Console.WriteLine("MakeSwaggerSDK - Swagger to C# SDK Generator");
            Console.WriteLine("============================================");

            return await Parser.Default.ParseArguments<Options>(args)
                .MapResult(async (Options opts) => await RunWithOptions(opts),
                          errs => Task.FromResult(1));
        }

        static async Task<int> RunWithOptions(Options options)
        {
            try
            {
                Console.WriteLine($"Parsing Swagger from: {options.SwaggerUrl}");
                Console.WriteLine($"SDK Name: {options.SdkName}");

                // Parse Swagger UI
                var parser = new SwaggerUiParser();
                var apiInfo = await parser.Parse(options.SwaggerUrl);

                if (apiInfo == null || apiInfo.Endpoints.Count == 0)
                {
                    Console.WriteLine("❌ No endpoints found or failed to parse Swagger documentation.");
                    return 1;
                }

                Console.WriteLine($"✅ Successfully parsed {apiInfo.Endpoints.Count} endpoints and {apiInfo.ClassDefinitions.Count} model classes.");

                // Generate SDK code
                var generator = new SwaggerClientCodeGenerator();
                var generatedCode = generator.Generate(options.SdkName, apiInfo);

                // Output the generated code
                var outputFileName = options.OutputFileName ?? $"{options.SdkName}.cs";
                await File.WriteAllTextAsync(outputFileName, generatedCode);

                Console.WriteLine($"✅ Generated SDK code saved to: {outputFileName}");
                
                Console.WriteLine("\nGenerated model classes:");
                foreach (var classDef in apiInfo.ClassDefinitions.Values)
                {
                    var classType = classDef.IsEnum ? "enum" : "class";
                    Console.WriteLine($"  - {classType} {classDef.Name} ({classDef.Properties.Count} properties)");
                }
                
                Console.WriteLine("\nGenerated endpoints:");
                foreach (var endpoint in apiInfo.Endpoints)
                {
                    Console.WriteLine($"  - {endpoint.HttpMethod.ToUpper()} {endpoint.Path} ({endpoint.OperationId})");
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                return 1;
            }
        }
    }
}
