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
            apiResponseClass.Properties.Should().BeEquivalentTo([
                new { Name = "code", Type = "int", IsRequired = true },
                new { Name = "type", Type = "string", IsRequired = true },
                new { Name = "message", Type = "string", IsRequired = true }
            ], options => options.ExcludingMissingMembers());
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
