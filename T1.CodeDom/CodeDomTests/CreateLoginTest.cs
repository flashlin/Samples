using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateLoginTest : TestBase
{
    public CreateLoginTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_login()
    {
        var sql = @"create login [user] with password = '123'";
        
        Parse(sql);

        ThenExprShouldBe(@"CREATE LOGIN [user] WITH PASSWORD = '123'");
    }
}