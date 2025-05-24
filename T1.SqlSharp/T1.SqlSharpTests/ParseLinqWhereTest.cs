using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseLinqWhereTest
{
    [Test]
    public void Should_parse_linq_with_where_clause()
    {
        // Arrange
        var linq = "from tb1 in test where tb1.Id == 1 select tb1";
        var linqParser = new LinqParser(linq);
        // Act
        var actual = linqParser.Parse();
        // Assert
        actual.ResultValue.ShouldBe(new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = "test",
                AliasName = "tb1"
            },
            Where = new LinqWhereExpr
            {
                Condition = new LinqConditionExpression
                {
                    Left = new LinqFieldExpr { TableOrAlias = "tb1", FieldName = "Id" },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new LinqValue { Value = "1" }
                }
            },
            Select = new LinqSelectAllExpr
            {
                AliasName = "tb1"
            }
        });
    }

    [Test]
    public void Should_parse_linq_with_multiple_where_conditions()
    {
        // Arrange
        var linq = "from tb1 in test where tb1.Id == 1 && tb1.Name == \"abc\" select tb1";
        var linqParser = new LinqParser(linq);
        // Act
        var actual = linqParser.Parse();
        // Assert
        actual.ResultValue.ShouldBe(new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = "test",
                AliasName = "tb1"
            },
            Where = new LinqWhereExpr
            {
                Condition = new LinqConditionExpression
                {
                    Left = new LinqConditionExpression
                    {
                        Left = new LinqFieldExpr { TableOrAlias = "tb1", FieldName = "Id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new LinqValue { Value = "1" }
                    },
                    LogicalOperator = LogicalOperator.And,
                    Right = new LinqConditionExpression
                    {
                        Left = new LinqFieldExpr { TableOrAlias = "tb1", FieldName = "Name" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new LinqValue { Value = "\"abc\"" }
                    }
                }
            },
            Select = new LinqSelectAllExpr
            {
                AliasName = "tb1"
            }
        });
    }
} 