using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class AlterTest : TestBase
{
    public AlterTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void alter_table()
    {
        var sql = "ALTER TABLE [dbo].[customer] ADD CONSTRAINT [DF_id]  DEFAULT ((0)) FOR [id]";
        Parse(sql);
        ThenExprShouldBe(@"ALTER TABLE [dbo].[customer] ADD CONSTRAINT [DF_id] DEFAULT ( 0 ) FOR [id]");
    }
}