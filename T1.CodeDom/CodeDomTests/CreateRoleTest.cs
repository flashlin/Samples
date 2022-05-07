using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateRoleTest : TestBase
{
    public CreateRoleTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_role()
    {
        var sql = @"create role [RoleName] AUTHORIZATION [dbo]";
        Parse(sql);

        ThenExprShouldBe(@"CREATE ROLE [RoleName] AUTHORIZATION [dbo]");
    }
}