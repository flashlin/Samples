using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseLinqJoinTest
{
    [Test]
    public void Should_parse_linq_with_join_clause()
    {
        // Arrange
        var linq = "from a in tableA join b in tableB on a.Id equals b.AId select a";
        var linqParser = new LinqParser(linq);
        // Act
        var actual = linqParser.Parse();
        // Assert
        actual.ResultValue.ShouldBe(new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = "tableA",
                AliasName = "a"
            },
            Joins =
            [
                new LinqJoinExpr
                {
                    AliasName = "b",
                    Source = "tableB",
                    On = new LinqConditionExpression
                    {
                        Left = new LinqFieldExpr { TableOrAlias = "a", FieldName = "Id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new LinqFieldExpr { TableOrAlias = "b", FieldName = "AId" }
                    }
                }
            ],
            Select = new LinqSelectAllExpr
            {
                AliasName = "a"
            }
        });
    }
}