using FluentAssertions;
using NSubstitute;
using T1.Standard.Serialization;
using WCodeSnippetX.Models;

namespace WCodeSnippetXTests;

public class CodeServiceTest
{
    private CodeSnippetService _service = null!;
    private JsonSerializer _jsonSerializer = null!;

    [SetUp]
    public void Setup()
    {
        var mockRepo = Substitute.For<ICodeSnippetRepo>();
        var codeSnippets = CreateCodeSnippetsData();
        mockRepo.QueryCode(Arg.Any<string>()).Returns(codeSnippets);

        _jsonSerializer = new JsonSerializer();
        _service = new CodeSnippetService(mockRepo, _jsonSerializer);
    }

    [Test]
    public void query_cs()
    {
        var actual = QueryCode("cs");

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


    [Test]
    public void query_class()
    {
        var actual = QueryCode("class");

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
                    Id = 3,
                    ProgramLanguage = "ts",
                    Content = "class User { }",
                    Description = "typescript"
                },
            }, opt => opt.WithStrictOrdering());
    }


    [Test]
    public void query_property()
    {
        var actual = QueryCode("property");

        actual.Should()
            .BeEquivalentTo(new[]
            {
                new CodeSnippetEntity()
                {
                    Id = 2,
                    ProgramLanguage = "cs",
                    Content = "public string Name { get; set; }",
                    Description = "property"
                },
            }, opt => opt.WithStrictOrdering());
    }


    [Test]
    public void query_empty()
    {
        var actual = QueryCode("");

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
                new CodeSnippetEntity()
                {
                    Id = 3,
                    ProgramLanguage = "ts",
                    Content = "class User { }",
                    Description = "typescript"
                },
            }, opt => opt.WithStrictOrdering());
    }

    [Test]
    public void query_cs_class()
    {
        var actual = QueryCode("cs class");

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
            }, opt => opt.WithStrictOrdering());
    }


    private List<CodeSnippetEntity> QueryCode(string text)
    {
        var actual = _jsonSerializer.Deserialize<List<CodeSnippetEntity>>(_service.Query(text));
        return actual;
    }

    private static List<CodeSnippetEntity> CreateCodeSnippetsData()
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
        return codeSnippets;
    }
}