using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateViewTest : TestBase
{
    public CreateViewTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_view()
    {
        var sql = @"create view [dbo].[vcustomer]
AS
SELECT id, name FROM customer WITH (NOLOCK)
ORDER BY name
";
        Parse(sql);

        ThenExprShouldBe(@"CREATE VIEW [dbo].[vcustomer]
AS
SELECT id, name FROM customer WITH( NOLOCK ) ORDER BY name ASC
");
    }
}