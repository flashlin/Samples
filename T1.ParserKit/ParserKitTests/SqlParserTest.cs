using FluentAssertions;
using ParserKitTests.Helpers;
using T1.ParserKit;

namespace ParserKitTests
{
    public class SqlParserTest
    {
        private SqlParser _sut = null!;

        [SetUp]
        public void SetUp()
        {
            _sut = new SqlParser();
        }

        [Test]
        public void Select()
        {
            var expr = _sut.Parse("select id from customer");

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
        public void SelectId_AliasName()
        {
            var expr = _sut.Parse("select id id1 from customer") as SelectExpr;

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

        [Test]
        public void SelectId_AS_AliasName()
        {
            var expr = _sut.Parse("select id as id1 from customer") as SelectExpr;

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
}