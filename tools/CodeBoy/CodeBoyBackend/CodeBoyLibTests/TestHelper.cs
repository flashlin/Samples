using CodeBoyLib.Models;
using CodeBoyLib.Services;
using System.Reflection;

namespace CodeBoyLibTests
{
    public static class TestHelper
    {
        public static string LoadEmbeddedResource(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            using var stream = assembly.GetManifestResourceStream(resourceName);
            
            if (stream == null)
            {
                throw new InvalidOperationException($"Could not find embedded resource: {resourceName}");
            }

            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        public static SwaggerApiInfo ParseSwaggerFromEmbeddedResource(string fileName)
        {
            var jsonContent = LoadEmbeddedResource($"CodeBoyLibTests.DataFiles.{fileName}");
            var parser = new SwaggerUiParser();
            return parser.ParseFromJson(jsonContent);
        }

        public static SwaggerApiInfo ParseSwaggerV2Json()
        {
            return ParseSwaggerFromEmbeddedResource("SwaggerV2.json");
        }

        public static SwaggerApiInfo ParseOpenApiV3Json()
        {
            return ParseSwaggerFromEmbeddedResource("OpenApiV3.json");
        }
    }
}

