using FluentAssertions;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharpTests;

[TestFixture]
public class StringParserTest
{
    [Test]
    public void SqlQuoted_Dot_Identifier()
    {
        var text = "[dbo].tb1 abc";
        var token = new StringParser(text).ReadSqlIdentifier();
        token.Word.Should().Be("[dbo].tb1");
    }
    
    
    [Test]
    public void ReadQuotedIdentifier()
    {
        var text = "[dbo].tb1 abc";
        var token = new StringParser(text).ReadQuotedIdentifier();
        token.Word.Should().Be("[dbo]");
    }
    
    [Test]
    public void ReadFullQuotedIdentifier()
    {
        var text = "[dbo].tb1 abc";
        var token = new StringParser(text).ReadFullQuotedIdentifier();
        token.Word.Should().Be("[dbo].tb1");
    }
}