using FluentAssertions;
using NSubstitute;
using T1.Standard.Serialization;
using WCodeSnippetX.Models;

namespace WCodeSnippetXTests;

public class CodeServiceTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void query_cs()
    {
        var codeSnippets = new List<CodeSnippetEntity>()
        {
            new CodeSnippetEntity()
            {
                Id = 1,
                ProgramLanguage = "cs",
                Content = "public class User { }",
                Description = "starter"
            },
            new CodeSnippetEntity()
            {
                Id = 2,
                ProgramLanguage = "cs",
                Content = "public string Name { get; set; }",
                Description = "property"
            },
            new CodeSnippetEntity()
            {
                Id = 3,
                ProgramLanguage = "ts",
                Content = "class User { }",
                Description = "typescript"
            },
        };

        var mockRepo = Substitute.For<ICodeSnippetRepo>();
        mockRepo.QueryCode(Arg.Any<string>()).Returns(codeSnippets);

        var jsonSerializer = new JsonSerializer();
        var service = new CodeSnippetService(mockRepo, jsonSerializer);

        var actual = jsonSerializer.Deserialize<List<CodeSnippetEntity>>(service.Query("cs"));

        actual.Should()
            .BeEquivalentTo(new[]
            {
                new CodeSnippetEntity()
                {
                    Id = 1,
                    ProgramLanguage = "cs",
                    Content = "public class User { }",
                    Description = "starter"
                },
                new CodeSnippetEntity()
                {
                    Id = 2,
                    ProgramLanguage = "cs",
                    Content = "public string Name { get; set; }",
                    Description = "property"
                },
            }, opt => opt.WithStrictOrdering());
    }
}