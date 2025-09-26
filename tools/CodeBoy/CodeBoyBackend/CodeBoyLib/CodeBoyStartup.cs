using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using CodeBoyLib.Services;

namespace CodeBoyLib
{
    /// <summary>
    /// Startup configuration for CodeBoy services
    /// </summary>
    public static class CodeBoyStartup
    {
        /// <summary>
        /// Adds CodeBoy services to the dependency injection container
        /// </summary>
        /// <param name="services">The service collection</param>
        /// <returns>The service collection for chaining</returns>
        public static IServiceCollection AddCodeBoyServices(this IServiceCollection services)
        {
            // Register database model generation services
            services.AddScoped<DatabaseModelGenerator>();
            services.AddScoped<GenDatabaseModelWorkflow>();
            
            // Register NuGet package generation service
            services.AddScoped<NupkgFileGenerator>();
            
            // Register Swagger/OpenAPI services
            services.AddScoped<SwaggerClientCodeGenerator>();
            services.AddScoped<SwaggerClientCsprojCodeGenerator>();
            services.AddScoped<GenSwaggerClientWorkflow>();
            services.AddScoped<SwaggerUiParser>();
            services.AddScoped<SwaggerDocumentDeserializer>();
            services.AddScoped<SwaggerV2ModelConverter>();
            services.AddScoped<SwaggerV3ModelConverter>();
            
            // Register project and build services
            services.AddScoped<CsprojService>();
            
            return services;
        }
    }
}
