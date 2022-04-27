using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateNonClusteredIndexTest : TestBase
{
    public CreateNonClusteredIndexTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void where()
    {
        var sql = @"CREATE NONCLUSTERED INDEX [IX_customer]
        ON [dbo].[customer]([id] ASC) WHERE ([id] IS NOT NULL) WITH (FILLFACTOR = 85);";
        
        Parse(sql);

        ThenExprShouldBe(@"CREATE NONCLUSTERED INDEX [IX_customer]
ON [dbo].[customer]([id] ASC)
WHERE ([id] IS NOT NULL) WITH(FILLFACTOR = 85) ;");
    }
}