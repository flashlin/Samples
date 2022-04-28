using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateClusteredIndexTest : TestBase
{
    public CreateClusteredIndexTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void nonclustered_where()
    {
        var sql = @"CREATE NONCLUSTERED INDEX [IX_customer]
        ON [dbo].[customer]([id] ASC) WHERE ([id] IS NOT NULL) WITH (FILLFACTOR = 85);";
        
        Parse(sql);

        ThenExprShouldBe(@"CREATE NONCLUSTERED INDEX [IX_customer]
ON [dbo].[customer]([id] ASC)
WHERE ([id] IS NOT NULL) WITH(FILLFACTOR = 85) ;");
    }
    
    
    [Fact]
    public void on_on()
    {
        var sql = @"CREATE CLUSTERED INDEX [ix_customer]
                       ON [dbo].[customer]([id] ASC, [name] ASC)
                       ON [product] ([ProductType]);";
        Parse(sql);
        ThenExprShouldBe(@"CREATE CLUSTERED INDEX [ix_customer]
ON [dbo].[customer]([id] ASC, [name] ASC)
ON [product]([ProductType])");
    }
    
    
    [Fact]
    public void unique_nonclustered()
    {
        var sql = @"CREATE UNIQUE NONCLUSTERED INDEX [IX_customer]
                        ON [dbo].[customer]([id] DESC) WITH (FILLFACTOR = 100);";
        Parse(sql);
        ThenExprShouldBe(@"CREATE UNIQUE NONCLUSTERED INDEX [IX_customer]
ON [dbo].[customer]([id] DESC) WITH(FILLFACTOR = 100) ;");
    }
    
    
}