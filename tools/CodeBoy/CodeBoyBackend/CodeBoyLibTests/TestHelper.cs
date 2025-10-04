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

        public static SwaggerApiInfo ParseSwaggerV2Json()
        {
            var swaggerV2JsonContent = LoadEmbeddedResource("CodeBoyLibTests.DataFiles.SwaggerV2.json");
            var parser = new SwaggerUiParser();
            return parser.ParseFromJson(swaggerV2JsonContent);
        }
    }
}

