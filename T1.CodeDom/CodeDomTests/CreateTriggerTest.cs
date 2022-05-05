using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateTriggerTest : TestBase
{
    public CreateTriggerTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_trigger()
    {
        var sql = @"create trigger myTrigger on database for customer
as
	set noexec on";
        Parse(sql);

        ThenExprShouldBe(@"CREATE TRIGGER myTrigger ON DATABASE FOR customer
AS
    SET NOEXEC ON");
    }
}