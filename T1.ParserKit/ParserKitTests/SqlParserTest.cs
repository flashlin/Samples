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
    public void Select()
    {
        var p = new SqlParser();
        var expr = p.Parse("select id from customer");
        expr.Should().BeEquivalentTo(
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
    
    [Test]
    public void SelectId()
    {
        var p = new SqlParser();
        var expr = p.Parse("select id id1 from customer");
        expr.Should().BeEquivalentTo(
            new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id",
                        AliasName = "id"
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