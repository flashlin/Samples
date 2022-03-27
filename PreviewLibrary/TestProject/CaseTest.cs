using ExpectedObjects;
using TestProject.Helpers;
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
		public void case_variable_when_constant_then()
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

		[Fact]
		public void case_when_then_arithmetic()
		{
			var sql = @"case @a
when 1 then @b + 1
else @b + 2
end";

			var expr = _sqlParser.ParseCasePartial(sql);

			@"CASE
	@a
	WHEN 1 THEN @b + 1
	ELSE @b + 2
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void case_func_when_number_then_func()
		{
			var sql = @"case datepart(DW, getdate()) 
			    when 2 then dateadd(dd, datediff(dd, 0,GETDATE()), 0 )
			    when 1 then dateadd(dd, datediff(dd, 0,GETDATE()-3), 0)
		    end";

			var expr = _sqlParser.ParseCasePartial(sql);
			@"CASE
	datepart( DW,getdate() )
	WHEN 2 THEN dateadd( dd,datediff( dd,0,GETDATE() ),0 )
	WHEN 1 THEN dateadd( dd,datediff( dd,0,GETDATE() - 3 ),0 )
END".ShouldEqual(expr);
		}

		[Fact]
		public void case_when_arithmetic_then_arithmetic()
		{
			var sql = @"case when (a + b) < (c - d) 
			then (a + d) end";

			var expr = _sqlParser.ParseCasePartial(sql);

			@"CASE
	WHEN (a + b) < (c - d) THEN a + d
END".ShouldEqual(expr);
		}

		[Fact]
		public void case_when()
		{
			var sql = "CASE WHEN @a = -1 THEN [b] ELSE @c END";
			var expr = _sqlParser.ParseCasePartial(sql);
			@"CASE
	WHEN @a = -1 THEN [b]
	ELSE @c
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}