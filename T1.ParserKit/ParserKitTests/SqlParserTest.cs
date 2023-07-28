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
        var exprs = p.Parse("select id from customer");
        exprs.Should().BeEquivalentTo(
            new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id"
                    }
                },
                FromClause = new TableExpr
                {
                    Name = "customer"
                }
            }
        );
    }
}