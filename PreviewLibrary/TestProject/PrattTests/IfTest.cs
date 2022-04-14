using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class IfTest : TestBase
	{
		public IfTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void if_begin_end()
		{
			var sql = @"if 'a' not like 'b'
begin
	set noexec on;
end";
			Parse(sql);

			ThenExprShouldBe(@"IF 'a' NOT LIKE 'b'
BEGIN
	SET NOEXEC ON 
	;
END");
		}

		[Fact]
		public void if_begin_end_else_begin_end()
		{
			var sql = @"if 'a' not like 'b'
begin
	set noexec on;
end else begin
	select 2
end";
			Parse(sql);

			ThenExprShouldBe(@"IF 'a' NOT LIKE 'b'
BEGIN
	SET NOEXEC ON
	;
END
ELSE BEGIN
	SELECT 2
END");
		}


	}
}
