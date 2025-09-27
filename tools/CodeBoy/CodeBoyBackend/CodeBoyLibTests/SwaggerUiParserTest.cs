using CodeBoyLib.Services;
using CodeBoyLib.Models;
using FluentAssertions;
using System.Linq;
using System.Reflection;
using NUnit.Framework;

namespace CodeBoyLibTests
{
    [TestFixture]
    public class SwaggerUiParserTest
    {
        [Test]
        public void ParseFromJson_WithSwaggerV2Json_ShouldParseApiResponseClassDefinition()
        {
            // Arrange
            var swaggerV2JsonContent = LoadEmbeddedResource("CodeBoyLibTests.DataFiles.SwaggerV2.json");
            var parser = new SwaggerUiParser();

            // Act
            var apiInfo = parser.ParseFromJson(swaggerV2JsonContent);

            // Assert
            apiInfo.Should().NotBeNull();
            apiInfo.ClassDefinitions.Should().NotBeNull();
            apiInfo.ClassDefinitions.Should().ContainKey("ApiResponse");

            var apiResponseClass = apiInfo.ClassDefinitions["ApiResponse"];
            apiResponseClass.Should().NotBeNull();
            apiResponseClass.Name.Should().Be("ApiResponse");

            // Verify ApiResponse has required properties
            var codeProperty = apiResponseClass.Properties.FirstOrDefault(p => p.Name == "code");
            codeProperty.Should().NotBeNull();
            codeProperty!.Type.Should().Be("int"); // Swagger V2 converter maps integer to int

            var typeProperty = apiResponseClass.Properties.FirstOrDefault(p => p.Name == "type");
            typeProperty.Should().NotBeNull();
            typeProperty!.Type.Should().Be("string");

            var messageProperty = apiResponseClass.Properties.FirstOrDefault(p => p.Name == "message");
            messageProperty.Should().NotBeNull();
            messageProperty!.Type.Should().Be("string");
        }

        private string LoadEmbeddedResource(string resourceName)
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
    }
}
