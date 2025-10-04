using CodeBoyLib.Models;
using FluentAssertions;
using System.Linq;
using NUnit.Framework;

namespace CodeBoyLibTests
{
    [TestFixture]
    public class SwaggerUiParserTest
    {
        [Test]
        public void ParseFromJson_WithSwaggerV2Json_ShouldParseApiResponseClassDefinition()
        {
            var apiInfo = TestHelper.ParseSwaggerV2Json();

            VerifyBasicStructure(apiInfo);
            VerifyApiResponseClass(apiInfo);
            VerifyPetClass(apiInfo);
        }

        private void VerifyBasicStructure(SwaggerApiInfo apiInfo)
        {
            apiInfo.Should().NotBeNull();
            apiInfo.ClassDefinitions.Should().NotBeNull();
        }

        private void VerifyApiResponseClass(SwaggerApiInfo apiInfo)
        {
            apiInfo.ClassDefinitions.Should().ContainKey("ApiResponse");

            var apiResponseClass = apiInfo.ClassDefinitions["ApiResponse"];
            apiResponseClass.Should().NotBeNull();
            apiResponseClass.Name.Should().Be("ApiResponse");

            apiResponseClass.Properties.Should().BeEquivalentTo([
                new { Name = "code", Type = "int", IsRequired = false },
                new { Name = "type", Type = "string", IsRequired = false },
                new { Name = "message", Type = "string", IsRequired = false }
            ], options => options.ExcludingMissingMembers());
        }

        private void VerifyPetClass(SwaggerApiInfo apiInfo)
        {
            apiInfo.ClassDefinitions.Should().ContainKey("Pet");

            var petClass = apiInfo.ClassDefinitions["Pet"];
            petClass.Should().NotBeNull();
            petClass.Name.Should().Be("Pet");

            petClass.Properties.Should().BeEquivalentTo([
                new { Name = "id", Type = "long", IsRequired = false },
                new { Name = "category", Type = "Category", IsRequired = false },
                new { Name = "name", Type = "string", IsRequired = true },
                new { Name = "photoUrls", Type = "List<string>", IsRequired = true },
                new { Name = "tags", Type = "List<Tag>", IsRequired = false },
                new { Name = "status", Type = "statusEnum", IsRequired = false }
            ], options => options.ExcludingMissingMembers());
        }
    }
}
