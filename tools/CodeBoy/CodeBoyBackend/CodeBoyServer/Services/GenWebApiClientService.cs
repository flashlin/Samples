using CodeBoyLib.Services;
using CodeBoyServer.Models;

namespace CodeBoyServer.Services
{
    /// <summary>
    /// Implementation of code generation service
    /// </summary>
    public class GenWebApiClientService : ICodeGenService
    {
        private readonly ILogger<GenWebApiClientService> _logger;

        public GenWebApiClientService(ILogger<GenWebApiClientService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generate Web API client code
        /// </summary>
        /// <param name="args">Generation arguments</param>
        /// <returns>Generated code</returns>
        public async Task<string> GenerateCode(GenWebApiClientArgs args)
        {
            try
            {
                _logger.LogInformation("Starting code generation for URL: {SwaggerUrl}, SDK: {SdkName}", 
                    args.SwaggerUrl, args.SdkName);

                // Parse Swagger UI
                var parser = new SwaggerUiParser();
                var apiInfo = await parser.ParseFromUrlAsync(args.SwaggerUrl);

                if (apiInfo == null || apiInfo.Endpoints.Count == 0)
                {
                    throw new InvalidOperationException("No endpoints found or failed to parse Swagger documentation.");
                }

                _logger.LogInformation("Successfully parsed {EndpointCount} endpoints and {ClassCount} model classes", 
                    apiInfo.Endpoints.Count, apiInfo.ClassDefinitions.Count);

                // Generate SDK code
                var generator = new SwaggerClientCodeGenerator();
                var generatedCode = generator.Generate(args.SdkName, apiInfo);

                _logger.LogInformation("Successfully generated code for SDK: {SdkName}", args.SdkName);

                return generatedCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating code for URL: {SwaggerUrl}, SDK: {SdkName}", 
                    args.SwaggerUrl, args.SdkName);
                throw;
            }
        }
    }
}
