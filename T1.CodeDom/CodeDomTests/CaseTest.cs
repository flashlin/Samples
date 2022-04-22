using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class CaseTest : TestBase
	{
		public CaseTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void case_when()
		{
			var sql = "case when @id = -1 then [Name] else @Name end";
			Parse(sql);
			ThenExprShouldBe("CASE WHEN @id = -1 THEN [Name] ELSE @Name END");
		}

		[Fact]
		public void case_when_case()
		{
			var sql = @"case
when @id in (1,2) then -- test
	case when (@sid in (3,4)) --test
		then 5
	else
		6
	end
when @id in (7,8)	then 9
else -1
end";
			Parse(sql);
			ThenExprShouldBe(@"CASE 
WHEN @id IN (1, 2) THEN 
	CASE WHEN ( @sid IN (3, 4) ) THEN 5 
	ELSE 6 
	END 
WHEN @id IN (7, 8) THEN 9 
ELSE -1
END");
		}

		[Fact]
		public void case_when_customFunc()
		{
			var sql = @"case when a > 1  and 
					[dbo].[fn_my] (1, 2) >= 2 
					then 2
					else 3 
				end";
			Parse(sql);

			ThenExprShouldBe(@"CASE WHEN a > 1 AND [dbo].[fn_my]( 1, 2 ) >= 2 THEN 2 ELSE 3 END");
		}

		[Fact]
		public void case_when_then_else_fn()
		{
			var sql = @"case when id=1 then '' else master.sys.fn_xxx(id) end ";
			Parse(sql);

			ThenExprShouldBe(@"CASE WHEN id = 1 THEN '' ELSE master.sys.fn_xxx( id ) END");
		}

		
	}
}
