using T1.SwaggerEx.Services;
using FluentAssertions;
using NUnit.Framework;

namespace T1.SwaggerExTests
{
    [TestFixture]
    public class SwaggerClientTypescriptCodeGeneratorTest
    {
        [Test]
        public void Generate_WithSwaggerV2Json_ShouldGenerateTypescriptCode()
        {
            var apiInfo = TestHelper.ParseSwaggerV2Json();
            var generator = new SwaggerClientTypescriptCodeGenerator();

            var result = generator.Generate("Petstore", apiInfo);

            VerifyBasicStructure(result);
            VerifyImportStatement(result);
            VerifyInterfacesGeneration(result);
            VerifyApiClientGeneration(result);
        }

        private void VerifyBasicStructure(string result)
        {
            result.Should().NotBeNullOrEmpty();
        }

        private void VerifyImportStatement(string result)
        {
            result.Should().Contain("import request from './request';");
        }

        private void VerifyInterfacesGeneration(string result)
        {
            result.Should().Contain("export interface ApiResponse {");
            result.Should().Contain("code?: number;");
            result.Should().Contain("type?: string;");
            result.Should().Contain("message?: string;");
            
            result.Should().Contain("export interface Pet {");
            result.Should().Contain("id?: number;");
            result.Should().Contain("name: string;");
            result.Should().Contain("photoUrls: string[];");
        }

        private void VerifyApiClientGeneration(string result)
        {
            result.Should().Contain("export const petstoreApi = {");
            result.Should().Contain("Promise<");
            result.Should().Contain("return request.");
        }
    }
}

