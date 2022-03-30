using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ReturnTest : SqlTestBase
	{
		public ReturnTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void return_none()
		{
			var sql = "return";
			
			var expr = _sqlParser.ParseReturnPartial(sql);
			
			"RETURN".ShouldEqual(expr);
		}

		[Fact]
		public void return_or()
		{
			var sql = "return @a | @b";

			var expr = _sqlParser.ParseReturnPartial(sql);

			"RETURN @a | @b".ShouldEqual(expr);
		}

		[Fact]
		public void return_case()
		{
			var sql = @"return case @a
when 1 then 2
end";

			var expr = _sqlParser.ParseReturnPartial(sql);

			"RETURN CASE @a WHEN 1 THEN 2 END"
				.ShouldEqual(expr);
		}

		[Fact]
		public void return_declare_var()
		{
			var sql = "return declare @a int = 1";

			var expr = _sqlParser.ParseReturnPartial(sql);

			"RETURN".ShouldEqual(expr);
		}
	}
}