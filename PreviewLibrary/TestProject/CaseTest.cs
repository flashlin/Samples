using ExpectedObjects;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class CaseTest : SqlTestBase
	{
		public CaseTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void case_when_a_in()
		{
			var sql = @"case when @a in (1,2) then select 1
else select 2
end";

			var expr = _sqlParser.ParseCasePartial(sql);

			@"CASE
	WHEN @a IN (1,2) THEN SELECT 1
	ELSE SELECT 2
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void case_when()
		{
			var sql = @"case @a
when 1 then select 2
else select 3
end            
";
			var expr = _sqlParser.ParseCasePartial(sql);

			@"CASE
	@a
	WHEN 1 THEN SELECT 2
	ELSE SELECT 3
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}