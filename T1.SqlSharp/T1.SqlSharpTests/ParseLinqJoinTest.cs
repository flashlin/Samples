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

    [Test]
    public void Should_parse_linq_with_left_join_clause()
    {
        // Arrange
        var linq = @"from c in customers 
                join o in orders on c.Id equals o.CustomerId into orderGroup 
                from og in orderGroup.DefaultIfEmpty() 
                select new { CustomerName = c.Name, OrderProduct = og?.Product ?? ""No Order"" }";
        var linqParser = new LinqParser(linq);
        // Act
        var actual = linqParser.Parse();
        // Assert
        actual.ResultValue.ShouldBe(new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = "customers",
                AliasName = "c"
            },
            Joins =
            [
                new LinqJoinExpr
                {
                    JoinType = "left join",
                    AliasName = "o",
                    Source = "orders",
                    On = new LinqConditionExpression
                    {
                        Left = new LinqFieldExpr { TableOrAlias = "c", FieldName = "Id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new LinqFieldExpr { TableOrAlias = "o", FieldName = "CustomerId" }
                    },
                    Into = "orderGroup"
                }
            ],
            AdditionalFroms =
            [
                new LinqFromExpr
                {
                    Source = "orderGroup.DefaultIfEmpty()",
                    AliasName = "og",
                    IsDefaultIfEmpty = true
                }
            ],
            Select = new LinqSelectNewExpr
            {
                Fields =
                [
                    new LinqSelectFieldExpr
                    {
                        Name = "CustomerName",
                        Value = new LinqFieldExpr { TableOrAlias = "c", FieldName = "Name" }
                    },
                    new LinqSelectFieldExpr
                    {
                        Name = "OrderProduct",
                        Value = new LinqValue { Value = "og?.Product ?? \"No Order\"" }
                    }
                ]
            }
        });
    }
}