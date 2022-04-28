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
    
    
    [Fact]
    public void alter_table_set()
    {
        var sql = "ALTER TABLE [customer] SET (LOCK_ESCALATION = AUTO);";
        Parse(sql);
        ThenExprShouldBe(@"ALTER TABLE [customer] SET (LOCK_ESCALATION = AUTO)");
    }
    
}