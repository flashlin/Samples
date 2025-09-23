using CodeBoyServer.Models;
using CodeBoyServer.Services;
using Microsoft.AspNetCore.Mvc;

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
        public static void GenWebApiClient(WebApplication app)
        {
            app.MapPost("/codegen/genWebApiClient", GenerateWebApiClient)
                .WithName("GenerateWebApiClient")
                .WithDescription("Generate Web API client code from Swagger URL")
                .WithTags("CodeGeneration")
                .WithOpenApi();
        }

        /// <summary>
        /// Generate Web API client endpoint handler
        /// </summary>
        /// <param name="request">Generation request</param>
        /// <param name="codeGenService">Code generation service</param>
        /// <param name="logger">Logger</param>
        /// <returns>Generated code response</returns>
        private static async Task<IResult> GenerateWebApiClient(
            [FromBody] GenWebApiClientRequest request,
            ICodeGenService codeGenService,
            ILogger<GenWebApiClientService> logger)
        {
            try
            {
                // Validate request
                if (string.IsNullOrWhiteSpace(request.SwaggerUrl))
                {
                    return Results.BadRequest(new GenWebApiClientResponse
                    {
                        Success = false,
                        ErrorMessage = "SwaggerUrl is required"
                    });
                }

                if (string.IsNullOrWhiteSpace(request.SdkName))
                {
                    return Results.BadRequest(new GenWebApiClientResponse
                    {
                        Success = false,
                        ErrorMessage = "SdkName is required"
                    });
                }

                logger.LogInformation("Received code generation request for URL: {SwaggerUrl}, SDK: {SdkName}", 
                    request.SwaggerUrl, request.SdkName);

                // Create generation arguments
                var args = new GenWebApiClientArgs
                {
                    SwaggerUrl = request.SwaggerUrl,
                    SdkName = request.SdkName
                };

                // Generate code
                var generatedCode = await codeGenService.GenerateCode(args);

                var response = new GenWebApiClientResponse
                {
                    Success = true,
                    GeneratedCode = generatedCode
                };

                logger.LogInformation("Successfully generated code for SDK: {SdkName}", request.SdkName);

                return Results.Ok(response);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error processing code generation request for URL: {SwaggerUrl}", 
                    request.SwaggerUrl);

                var errorResponse = new GenWebApiClientResponse
                {
                    Success = false,
                    ErrorMessage = ex.Message
                };

                return Results.Problem(
                    title: "Code Generation Error",
                    detail: ex.Message,
                    statusCode: 500);
            }
        }
    }
}
