using CodeBoyServer.Models;
using CodeBoyServer.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
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
            
            app.MapPost("/api/codegen/buildDatabaseModelNupkg", BuildDatabaseModelNupkg)
                .WithDescription("Build database model nupkg from database connection")
                .WithTags("CodeGeneration")
                .WithOpenApi();
            
            app.MapPost("/api/codegen/genTypescriptCodeFromSwagger", GenTypescriptCodeFromSwagger)
                .WithDescription("Generate TypeScript API client code from Swagger URL")
                .WithTags("CodeGeneration")
                .WithOpenApi();
            
            app.MapPost("/api/codegen/genDatabaseDto", GenDatabaseDto)
                .WithDescription("Generate database DTO code from SQL CREATE TABLE statement")
                .WithTags("CodeGeneration")
                .WithOpenApi();
            
            app.MapPost("/api/codegen/genProtoCodeFromGrpcClientAssembly", GenProtoCodeFromGrpcClientAssembly)
                .WithDescription("Generate proto code from gRPC client assembly")
                .WithTags("CodeGeneration")
                .DisableAntiforgery()
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
            var args = new GenWebApiClientArgs
            {
                SwaggerUrl = request.SwaggerUrl,
                SdkName = request.SdkName
            };
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
            // Set up output directory 
            var outputPath = Path.Combine(Path.GetTempPath(), $"CodeBoy_{Guid.NewGuid():N}");
            Directory.CreateDirectory(outputPath);

            // Build the Swagger client using GenSwaggerClientWorkflow
            var workflow = new GenSwaggerClientWorkflow();
            var buildParams = new GenSwaggerClientBuildParams
            {
                SdkName = request.SdkName,
                SwaggerUrl = request.SwaggerUrl,
                OutputPath = outputPath,
                NupkgName = request.NupkgName,
                SdkVersion = request.SdkVersion
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

        private static async Task<IResult> BuildDatabaseModelNupkg(
            [FromBody] BuildDatabaseModelNupkgRequest request,
            IGenDatabaseModelWorkflow workflow)
        {
            // Set up output directory 
            var outputPath = Path.Combine(Path.GetTempPath(), $"CodeBoy_DB_{Guid.NewGuid():N}");
            Directory.CreateDirectory(outputPath);

            // Build the database model using GenDatabaseModelWorkflow (injected)
            var buildParams = new GenDatabaseModelBuildParams
            {
                DatabaseServer = request.DatabaseServer,
                LoginId = request.LoginId,
                LoginPassword = request.LoginPassword,
                DatabaseName = request.DatabaseName,
                NamespaceName = request.NamespaceName,
                SdkName = request.SdkName,
                SdkVersion = request.SdkVersion,
                TargetFrameworks = request.TargetFrameworks,
                OutputPath = outputPath
            };

            var result = await workflow.Build(buildParams);

            if (!result.Success)
            {
                var errorMessage = result.Errors.Any() 
                    ? string.Join("; ", result.Errors)
                    : "Unknown error occurred during database model build process";
                return Results.Problem(
                    detail: errorMessage,
                    statusCode: 500,
                    title: "Database Model Build Failed"
                );
            }

            // Check if NuGet package was generated
            if (string.IsNullOrEmpty(result.NupkgFile) || !File.Exists(result.NupkgFile))
            {
                return Results.Problem(
                    detail: $"Generated .nupkg file not found at: {result.NupkgFile}",
                    statusCode: 500,
                    title: "NuGet Package Not Found"
                );
            }

            // Read the file for download
            var fileBytes = await File.ReadAllBytesAsync(result.NupkgFile);
            var fileName = Path.GetFileName(result.NupkgFile);

            // Return file download response
            return Results.File(
                fileContents: fileBytes,
                contentType: "application/octet-stream",
                fileDownloadName: fileName
            );
        }

        /// <summary>
        /// Generate TypeScript code from Swagger endpoint handler
        /// </summary>
        /// <param name="request">TypeScript generation request</param>
        /// <returns>Generated TypeScript code</returns>
        private static async Task<string> GenTypescriptCodeFromSwagger(
            [FromBody] GenTypescriptCodeFromSwaggerRequest request)
        {
            var swaggerParser = new SwaggerUiParser();
            var apiInfo = await swaggerParser.ParseFromJsonUrlAsync(request.SwaggerUrl);

            var typescriptGenerator = new SwaggerClientTypescriptCodeGenerator();
            return typescriptGenerator.Generate(request.ApiName, apiInfo);
        }

        /// <summary>
        /// Generate database DTO code from SQL CREATE TABLE statement
        /// </summary>
        /// <param name="request">Database DTO generation request</param>
        /// <returns>Generated DTO code</returns>
        private static string GenDatabaseDto([FromBody] GenDatabaseDtoRequest request)
        {
            var generator = new DatabaseDtoGenerator();
            return generator.GenerateEfDtoCode(request.Sql);
        }

        private static async Task<List<ProtoFileInfo>> GenProtoCodeFromGrpcClientAssembly(
            [FromForm] string namespaceName,
            [FromForm] IFormFile assemblyFile,
            ILogger<Program> logger)
        {
            if (assemblyFile == null || assemblyFile.Length == 0)
            {
                throw new ArgumentException("Assembly file is required");
            }

            byte[] assemblyBytes;
            using (var memoryStream = new MemoryStream())
            {
                await assemblyFile.CopyToAsync(memoryStream);
                assemblyBytes = memoryStream.ToArray();
            }

            var appDirectory = AppDomain.CurrentDomain.BaseDirectory;
            var dependenciesDirectory = Path.Combine(appDirectory, "codeData");
            
            if (!Directory.Exists(dependenciesDirectory))
            {
                Directory.CreateDirectory(dependenciesDirectory);
            }

            var generator = new GrpcSdkWarpGenerator(logger);
            var types = generator.QueryGrpcClientTypesFromAssemblyBytes(assemblyBytes, dependenciesDirectory);
            
            var protoFiles = new List<ProtoFileInfo>();
            foreach (var type in types)
            {
                var genProtoCode = generator.GenProtoCode(type);
                protoFiles.Add(new ProtoFileInfo()
                {
                    ServiceName = type.Name,
                    ProtoCode = genProtoCode,
                });
            }
            return protoFiles;
        }
    }

    public class ProtoFileInfo
    {
        public string ServiceName { get; set; } = string.Empty;
        public string ProtoCode { get; set; } =string.Empty;
    }
}
