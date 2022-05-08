using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateUserTest : TestBase
{
    public CreateUserTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_user()
    {
        var sql = @"create user [UserName] FOR LOGIN [LoginName] WITH DEFAULT_SCHEMA=[dbo]";
        Parse(sql);

        ThenExprShouldBe(@"CREATE USER [UserName] FOR LOGIN [LoginName] WITH DEFAULT_SCHEMA = [dbo]");
    }
    
    
    [Fact]
    public void create_user_without()
    {
        var sql = @"create user [UserName] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]";
        Parse(sql);

        ThenExprShouldBe(@"CREATE USER [UserName] WITHOUT LOGIN WITH DEFAULT_SCHEMA = [dbo]");
    }
    
    
}