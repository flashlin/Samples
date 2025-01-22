using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.ParserLit;

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

    [Test]
    public void Read_table_dot_star()
    {
        var text = "table.* abc";
        var token = new StringParser(text).ReadFullQuotedIdentifier();
        token.Word.Should().Be("table.*");
    }
    
    [Test]
    public void Read_dot_dot_identifier()
    {
        var text = "[db]..field abc";
        var token = new StringParser(text).ReadFullQuotedIdentifier();
        token.Word.Should().Be("[db]..field");
    }

    [Test]
    public void ReadSymbols()
    {
        var text = "/*123*/";
        var token = new StringParser(text).ReadSymbols();
        token.Word.Should().Be("/*");
    }
    
    [Test]
    public void ReadStarComment()
    {
        var text = "/*123*/abc";
        var token = new StringParser(text).ReadDoubleComment();
        token.Word.Should().Be("/*123*/");
    }
}