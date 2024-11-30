using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using SqlSharpLit.Common.ParserLit.Expressions;
using T1.SqlSharp.Expressions;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

public static class TestHelper
{
    public static void ShouldBe<T>(this ParseResult<ISqlExpression> rc, T expectedSqlStatement)
        where T : ISqlExpression
    {
        if (rc.HasError)
        {
            throw new Exception(rc.Error.Message);
        }
        
        var castedStatement = (T)rc.Result;
        castedStatement.Should().BeEquivalentTo(expectedSqlStatement);
    }
}