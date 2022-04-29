using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class CreateProcedureTest : TestBase
{
    public CreateProcedureTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void create_procedure_no_parameters()
    {
        var sql = @"create procedure myProc
as
begin
	EXEC [db].[dbo].[fn_my]
end
go";
        Parse(sql);

        ThenExprShouldBe(@"CREATE PROCEDURE myProc
AS
BEGIN
	EXEC [db].[dbo].[fn_my]
END
GO");
    }
}