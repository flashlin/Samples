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
                new { Name = "code", Type = "int", IsRequired = false },
                new { Name = "type", Type = "string", IsRequired = false },
                new { Name = "message", Type = "string", IsRequired = false }
            ], options => options.ExcludingMissingMembers());

            // Verify Pet class definition exists
            apiInfo.ClassDefinitions.Should().ContainKey("Pet");
            var petClass = apiInfo.ClassDefinitions["Pet"];
            petClass.Should().NotBeNull();
            petClass.Name.Should().Be("Pet");

            // Verify Pet has required properties with correct IsRequired values
            petClass.Properties.Should().BeEquivalentTo([
                new { Name = "id", Type = "long", IsRequired = false },
                new { Name = "category", Type = "Category", IsRequired = false },
                new { Name = "name", Type = "string", IsRequired = true },
                new { Name = "photoUrls", Type = "List<string>", IsRequired = true },
                new { Name = "tags", Type = "List<Tag>", IsRequired = false },
                new { Name = "status", Type = "statusEnum", IsRequired = false }
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
