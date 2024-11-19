using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

public static class TestHelper
{
    public static void ShouldBe<T>(this Either<ISqlExpression, ParseError> rc, T expectedSqlStatement)
        where T : ISqlExpression
    {
        rc.Switch(statement =>
            {
                var castedStatement = (T)statement;
                castedStatement.Should().BeEquivalentTo(expectedSqlStatement);
            },
            error => throw error
        );
    }
}