using FluentAssertions;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ExtractStatementResultsTest
{
    [Test]
    public void ExtractStatementResults_ShouldReturnStatementsUntilFirstError()
    {
        var sql = """
                  select 1
                  dump database mydb
                  select 2
                  """;

        var results = new SqlParser(sql)
            .ExtractStatementResults()
            .Select(x => new
            {
                x.HasError,
                TypeName = x.HasError ? string.Empty : x.ResultValue.GetType().Name,
                ErrorMessage = x.HasError ? x.Error.Message : string.Empty
            })
            .ToList();

        results.Should().BeEquivalentTo([
            new
            {
                HasError = false,
                TypeName = nameof(SelectStatement),
                ErrorMessage = string.Empty
            },
            new
            {
                HasError = true,
                TypeName = string.Empty,
                ErrorMessage = "Unknown statement"
            }
        ]);
    }
}
