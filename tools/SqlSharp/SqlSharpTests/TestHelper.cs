using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp;
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
        
        var castedStatement = (T)rc.ResultValue;
        castedStatement.Should().BeEquivalentTo(expectedSqlStatement, 
            options => options.RespectingRuntimeTypes()
                .WithTracing()
                .ExcludingMissingMembers()
                .Using<TextSpan>(_ => { })
                .WhenTypeIs<TextSpan>()
                .Excluding(x => x.Span)
            );
    }
    
    public static void ShouldBe<T>(this object rc, T expected)
    {
        rc.Should().BeEquivalentTo(expected, 
            options => options.RespectingRuntimeTypes()
                .WithTracing()
                .Using<TextSpan>(_ => { })
                .WhenTypeIs<TextSpan>()
            );
    }

    public static ParseResult<ISqlExpression> ParseSql(this string text)
    {
        var sqlParser = new SqlParser(text);
        return sqlParser.Parse();
    }
}