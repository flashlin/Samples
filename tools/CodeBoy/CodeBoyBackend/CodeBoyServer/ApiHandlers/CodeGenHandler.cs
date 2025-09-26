using CodeBoyServer.Models;
using CodeBoyServer.Services;
using Microsoft.AspNetCore.Mvc;
using CodeBoyLib.Services;
using CodeBoyLib.Models;

namespace CodeBoyServer.ApiHandlers
{
    /// <summary>
    /// Handler for code generation API endpoints
    /// </summary>
    public static class CodeGenHandler
    {
        /// <summary>
        /// Configure code generation endpoints
        /// </summary>
        /// <param name="app">Web application builder</param>
        public static void Map(WebApplication app)
        {
            app.MapPost("/api/codegen/genWebApiClient", GenerateWebApiClient)
                .WithName("GenerateWebApiClient")
                .WithDescription("Generate Web API client code from Swagger URL")
                .WithTags("CodeGeneration")
                .WithOpenApi();
            
            app.MapPost("/api/codegen/buildWebApiClientNupkg", BuildWebApiClientNupkg)
                .WithDescription("Build Web API client nupkg from Swagger URL")
                .WithTags("CodeGeneration")
                .WithOpenApi();
        }

        /// <summary>
        /// Generate Web API client endpoint handler
        /// </summary>
        /// <param name="request">Generation request</param>
        /// <param name="codeGenService">Code generation service</param>
        /// <returns>Generated code</returns>
        private static async Task<string> GenerateWebApiClient(
            [FromBody] GenWebApiClientRequest request,
            ICodeGenService codeGenService)
        {
            // Validate request
            if (string.IsNullOrWhiteSpace(request.SwaggerUrl))
            {
                throw new ArgumentException("SwaggerUrl is required");
            }

            if (string.IsNullOrWhiteSpace(request.SdkName))
            {
                throw new ArgumentException("SdkName is required");
            }

            // Create generation arguments
            var args = new GenWebApiClientArgs
            {
                SwaggerUrl = request.SwaggerUrl,
                SdkName = request.SdkName
            };

            // Generate and return code
            return await codeGenService.GenerateCode(args);
        }

        /// <summary>
        /// Build Web API client nupkg endpoint handler
        /// </summary>
        /// <param name="request">Build request</param>
        /// <returns>File download response for the generated .nupkg file</returns>
        private static async Task<IResult> BuildWebApiClientNupkg(
            [FromBody] BuildWebApiClientNupkgRequest request)
        {
            // Validate request
            if (string.IsNullOrWhiteSpace(request.SdkName))
            {
                return Results.BadRequest("SdkName is required");
            }

            if (string.IsNullOrWhiteSpace(request.SwaggerUrl))
            {
                return Results.BadRequest("SwaggerUrl is required");
            }

            if (string.IsNullOrWhiteSpace(request.NupkgName))
            {
                return Results.BadRequest("NupkgName is required");
            }

            // Parse Swagger API information from URL
            var swaggerParser = new SwaggerUiParser();
            var apiInfo = await swaggerParser.ParseFromJsonUrlAsync(request.SwaggerUrl);

            // Set up output directory 
            var outputPath = Path.Combine(Path.GetTempPath(), $"CodeBoy_{Guid.NewGuid():N}");
            Directory.CreateDirectory(outputPath);

            // Build the Swagger client using GenSwaggerClientWorkflow
            var workflow = new GenSwaggerClientWorkflow();
            var buildParams = new GenSwaggerClientBuildParams
            {
                SdkName = request.SdkName,
                ApiInfo = apiInfo,
                OutputPath = outputPath,
                NupkgName = request.NupkgName,
                SdkVersion = "1.0.0" // Default version, could be configurable
            };
            var result = await workflow.Build(buildParams);

            if (!result.Success)
            {
                var errorMessage = result.Errors.Any() 
                    ? string.Join("; ", result.Errors)
                    : "Unknown error occurred during build process";
                return Results.Problem(
                    detail: errorMessage,
                    statusCode: 500,
                    title: "Build Failed"
                );
            }

            // Find the generated .nupkg file
            var nupkgFile = Path.Combine(outputPath, $"{buildParams.NupkgName}.{buildParams.SdkVersion}.nupkg");
            if (!File.Exists(nupkgFile))
            {
                return Results.Problem(
                    detail: $"Generated .nupkg file not found at: {nupkgFile}",
                    statusCode: 500,
                    title: "File Not Found"
                );
            }

            // Read the file for download
            var fileBytes = await File.ReadAllBytesAsync(nupkgFile);
            var fileName = Path.GetFileName(nupkgFile);

            // Return file download response
            return Results.File(
                fileContents: fileBytes,
                contentType: "application/octet-stream",
                fileDownloadName: fileName
            );
        }
    }
}
