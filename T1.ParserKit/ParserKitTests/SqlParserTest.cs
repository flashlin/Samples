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
        var sut = new SqlParser();
        
        var expr = sut.Parse("select id id1 from customer") as SelectExpr;
        
        var expectedExpr = new SelectExpr
        {
            Columns = new List<SqlExpr>()
            {
                new FieldExpr
                {
                    Name = "id",
                    AliasName = "id1"
                },
            },
            FromClause = new TableExpr
            {
                Name = "customer",
                AliasName = string.Empty
            }
        };
        
        expr.Should().BeEquivalentTo(
            expectedExpr
        );

        expr!.Columns.AllSatisfy(expectedExpr.Columns);
    }
}

public static class FluentAssertionsExtensions
{
    public static void AllSatisfy<T>(this ICollection<T> items, ICollection<T> expectedItems)
    {
        foreach (var (item, expected) in items.Zip(expectedItems))
        {
            item.Should().BeOfType(expected!.GetType())
                .And.BeEquivalentTo(expected,
                    options => options.IncludingAllRuntimeProperties());
        }
    }
}