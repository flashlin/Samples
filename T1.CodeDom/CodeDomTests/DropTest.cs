using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class DropTest : TestBase
{
    public DropTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }
        
    [Fact]
    public void drop_role()
    {
        var sql = @"drop role roleName";

        Parse(sql);

        ThenExprShouldBe(@"DROP ROLE roleName");
    }
    
    
    [Fact]
    public void drop_trigger()
    {
        var sql = @"drop trigger [tr_customer]";

        Parse(sql);

        ThenExprShouldBe(@"DROP TRIGGER [tr_customer]");
    }
}