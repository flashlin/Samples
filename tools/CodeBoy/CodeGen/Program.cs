using CommandLine;
using CodeBoyLib.Models;
using CodeBoyLib.Services;
using System;
using System.Collections.Generic;
using System.IO;
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

        [Option('p', "output-path", Required = true, HelpText = "Base output path for all build operations")]
        public string OutputPath { get; set; } = string.Empty;
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
                if (!string.IsNullOrEmpty(options.OutputPath))
                {
                    Console.WriteLine($"Output Path: {options.OutputPath}");
                }

                // Parse Swagger UI
                var parser = new SwaggerUiParser();
                var apiInfo = await parser.ParseFromJsonUrlAsync(options.SwaggerUrl);

                if (apiInfo == null || apiInfo.Endpoints.Count == 0)
                {
                    Console.WriteLine("❌ No endpoints found or failed to parse Swagger documentation.");
                    return 1;
                }

                Console.WriteLine($"✅ Successfully parsed {apiInfo.Endpoints.Count} endpoints and {apiInfo.ClassDefinitions.Count} model classes.");

                // Use the specified output path directly
                var outputPath = options.OutputPath;
                Console.WriteLine($"📁 Using output path: {outputPath}");
                
                // Ensure output directory exists
                Directory.CreateDirectory(outputPath);

                // Generate and build complete SDK project
                var factory = new GenSwaggerClientWorkflow();
                var result = await factory.Build(options.SdkName, apiInfo, outputPath, options.SdkName);

                // Print detailed summary
                factory.PrintSummary(result);

                if (result.Success)
                {
                    // Copy the generated client code to the requested output location if specified
                    if (!string.IsNullOrEmpty(options.OutputFileName))
                    {
                        File.Copy(result.ClientCodePath, options.OutputFileName, true);
                        Console.WriteLine($"📋 Also copied client code to: {options.OutputFileName}");
                    }

                    Console.WriteLine($"\n🎉 Complete SDK project generated successfully!");
                    Console.WriteLine($"📦 Assembly: {result.AssemblyPath}");
                    Console.WriteLine($"📁 Project files: {result.TempDirectory}");
                    
                    return 0;
                }
                else
                {
                    Console.WriteLine($"\n❌ SDK generation failed. Check errors above.");
                    return 1;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                return 1;
            }
        }
    }
}
