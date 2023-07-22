using FluentAssertions;
using T1.ParserKit;

namespace ParserKitTests;

public class SqlParserTest
{
    [SetUp]
    public void SetUp()
    {
    }

    [Test]
    public void Where()
    {
        var p = new SqlParser();
        var exprs = p.Parse("where id=1")
            .ToArray();
        exprs.Should().BeEquivalentTo(new[]
        {
            new WhereExpr()
        });
    }
}