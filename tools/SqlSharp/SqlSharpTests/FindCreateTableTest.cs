using FluentAssertions;
using NSubstitute;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharpTests;

[TestFixture]
public class FindCreateTableTest
{
    [Test]
    public void METHOD()
    {
        var text = $"""
                    --create table #directcustomerid (custid int, username nvarchar(50));
                    """;
        var databaseNameProvider = Substitute.For<IDatabaseNameProvider>();
        var rc = new ExtractSqlHelper(databaseNameProvider)
            .ExtractCreateTableFromText(text);
        rc.Should().BeEquivalentTo((string.Empty,string.Empty));
    }
}