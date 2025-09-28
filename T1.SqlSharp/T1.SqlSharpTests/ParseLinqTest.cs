using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseLinqTest
{
    [Test]
    public void Should_parse_simple_linq_from_select()
    {
        // Arrange
        var linq = "from tb1 in test select tb1";
        var linqParser = new LinqParser(linq);
        // Act
        var actual = linqParser.Parse();
        // Assert
        actual.ResultValue.ShouldBe(new LinqExpr
        {
            From = new LinqFromExpr
            {
                Source = new LinqSourceExpr { TableName = "test" },
                AliasName = "tb1"
            },
            Select = new LinqSelectAllExpr
            {
                AliasName = "tb1"
            }
        });
    }
} 