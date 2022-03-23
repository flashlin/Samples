using ExpectedObjects;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class IfTest : SqlTestBase
	{
		public IfTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void if_begin_end_else_begin_end()
		{
			var sql = @"if @id = 1
begin select 2 end
else begin select 3 end";
			var expr = _sqlParser.ParseIfPartial(sql);
			@"IF @id = 1
BEGIN
SELECT 2
END
ELSE BEGIN
SELECT 3
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void if_a_not_like_b()
		{
			var sql = @"IF N'a' NOT LIKE N'True'
BEGIN
	select 1
END";
			var expr = _sqlParser.ParseIfPartial(sql);
			@"IF N'a' NOT LIKE N'True'
BEGIN
SELECT 1
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}