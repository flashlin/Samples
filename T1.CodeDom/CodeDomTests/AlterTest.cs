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
    public void alter_table_add_columns()
    {
        var sql = @"ALTER TABLE [dbo].[customer] 
ADD Addr char(10),Tel int";
        Parse(sql);
        ThenExprShouldBe(@"ALTER TABLE [dbo].[customer] ADD Addr CHAR (10), Tel INT");
    }
    
    
    [Fact]
    public void alter_table_set()
    {
        var sql = "ALTER TABLE [customer] SET (LOCK_ESCALATION = AUTO);";
        Parse(sql);
        ThenExprShouldBe(@"ALTER TABLE [customer] SET (LOCK_ESCALATION = AUTO)");
    }
    
    [Fact]
    public void alter_table_primary_key()
    {
        var sql = @"ALTER TABLE [customer] 
	ADD CONSTRAINT [PK_customer] PRIMARY KEY CLUSTERED ([Id] )";
        Parse(sql);
        ThenExprShouldBe(@"ALTER TABLE [customer] ADD CONSTRAINT [PK_customer] PRIMARY KEY CLUSTERED([Id] ASC)");
    }
    
    
    [Fact]
    public void alter_index()
    {
        var sql = @"ALTER index all ON customer REBUILD WITH (FILLFACTOR = 90, online=on);";
        Parse(sql);
        ThenExprShouldBe(@"ALTER INDEX ALL ON customer REBUILD WITH(FILLFACTOR = 90, ONLINE = ON)");
    }
}