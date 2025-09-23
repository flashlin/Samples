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
        public static void Map(WebApplication app)
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
    }
}
