using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateTableTest : TestBase
{
    public CreateTableTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_table_not_for_replication()
    {
        var sql = @"CREATE TABLE [dbo].[customer] (
    [id] int NOT FOR REPLICATION NOT NULL
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT FOR REPLICATION NOT NULL
) ;");
    }
    
    
    [Fact]
    public void create_table_primary_key_clustered()
    {
        var sql = @"create table [dbo].[customer] (
    [id]      INT            NOT NULL,
    PRIMARY KEY CLUSTERED ([id] ASC)
);";

        Parse(sql);

        ThenExprShouldBe(@"CREATE TABLE [dbo].[customer](
    [id] INT NOT NULL,
    PRIMARY KEY CLUSTERED([id] ASC)
) ;");
    }
    
    
}