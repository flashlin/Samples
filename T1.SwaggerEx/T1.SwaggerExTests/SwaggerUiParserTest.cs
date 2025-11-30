using T1.SwaggerEx.Models;
using FluentAssertions;
using System.Linq;
using NUnit.Framework;

namespace T1.SwaggerExTests
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

            abTestRequest.Properties.Should().BeEquivalentTo([
                new { Name = "Name", Type = "string", IsRequired = false },
                new { Name = "From", Type = "string", IsRequired = false },
                new { Name = "Group", Type = "string", IsRequired = false },
                new { Name = "Platform", Type = "string", IsRequired = false },
                new { Name = "Language", Type = "string", IsRequired = false },
                new { Name = "Event", Type = "string", IsRequired = false },
                new { Name = "Category", Type = "string", IsRequired = false },
                new { Name = "LoginName", Type = "string", IsRequired = false }
            ], options => options.ExcludingMissingMembers());
        }

        private void VerifyRegistrationFormClass(SwaggerApiInfo apiInfo)
        {
            apiInfo.ClassDefinitions.Should().ContainKey("RegistrationForm");

            var registrationForm = apiInfo.ClassDefinitions["RegistrationForm"];
            registrationForm.Should().NotBeNull();
            registrationForm.Name.Should().Be("RegistrationForm");

            registrationForm.Properties.Should().BeEquivalentTo([
                new { Name = "AccountInfo", Type = "AccountInfo", IsRequired = false },
                new { Name = "PersonalInfo", Type = "PersonalInfo", IsRequired = false },
                new { Name = "Platform", Type = "string", IsRequired = false },
                new { Name = "Language", Type = "int", IsRequired = false },
                new { Name = "SecurityAnswer", Type = "string", IsRequired = false },
                new { Name = "SecurityQuestion", Type = "string", IsRequired = false },
                new { Name = "Referral", Type = "string", IsRequired = false },
                new { Name = "SecurityQuestionId", Type = "int", IsRequired = false },
                new { Name = "ValidationCodeLoginName", Type = "string", IsRequired = false },
                new { Name = "ValidationCode", Type = "string", IsRequired = false },
                new { Name = "TCReadTime", Type = "string", IsRequired = false },
                new { Name = "BrandRedirection", Type = "RegisterBrandRedirection", IsRequired = false },
                new { Name = "IsRemoveNameAndDob", Type = "bool", IsRequired = false }
            ], options => options.ExcludingMissingMembers());
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
