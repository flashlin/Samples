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

        [Test]
        public void ParseFromJson_WithOpenApiV3Json_ShouldParseCorrectly()
        {
            var apiInfo = TestHelper.ParseOpenApiV3Json();

            VerifyBasicStructure(apiInfo);
            VerifyOpenApiV3BasicInfo(apiInfo);
            VerifyAbTestRequestClass(apiInfo);
            VerifyRegistrationFormClass(apiInfo);
        }

        private void VerifyBasicStructure(SwaggerApiInfo apiInfo)
        {
            apiInfo.Should().NotBeNull();
            apiInfo.ClassDefinitions.Should().NotBeNull();
        }

        private void VerifyOpenApiV3BasicInfo(SwaggerApiInfo apiInfo)
        {
            apiInfo.Title.Should().Be("Akis");
            apiInfo.Version.Should().Be("1.0");
            apiInfo.Endpoints.Should().NotBeEmpty();
            apiInfo.ClassDefinitions.Should().NotBeEmpty();
        }

        private void VerifyAbTestRequestClass(SwaggerApiInfo apiInfo)
        {
            apiInfo.ClassDefinitions.Should().ContainKey("AbTestRequest");

            var abTestRequest = apiInfo.ClassDefinitions["AbTestRequest"];
            abTestRequest.Should().NotBeNull();
            abTestRequest.Name.Should().Be("AbTestRequest");

            abTestRequest.Properties.Should().Contain(p => p.Name == "Name" && p.Type == "string");
            abTestRequest.Properties.Should().Contain(p => p.Name == "From" && p.Type == "string");
            abTestRequest.Properties.Should().Contain(p => p.Name == "Group" && p.Type == "string");
            abTestRequest.Properties.Should().Contain(p => p.Name == "Platform" && p.Type == "string");
        }

        private void VerifyRegistrationFormClass(SwaggerApiInfo apiInfo)
        {
            apiInfo.ClassDefinitions.Should().ContainKey("RegistrationForm");

            var registrationForm = apiInfo.ClassDefinitions["RegistrationForm"];
            registrationForm.Should().NotBeNull();
            registrationForm.Name.Should().Be("RegistrationForm");

            registrationForm.Properties.Should().Contain(p => p.Name == "AccountInfo" && p.Type == "AccountInfo");
            registrationForm.Properties.Should().Contain(p => p.Name == "PersonalInfo" && p.Type == "PersonalInfo");
            registrationForm.Properties.Should().Contain(p => p.Name == "Platform" && p.Type == "string");
            registrationForm.Properties.Should().Contain(p => p.Name == "Language");
            registrationForm.Properties.Should().Contain(p => p.Name == "SecurityQuestionId" && p.Type == "int");
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
