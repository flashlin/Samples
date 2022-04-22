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

		[Fact]
		public void if_else_if_else()
		{
			var sql = @"if @id in (1,2) --test
           set @r = 0
        else if @id = 2 -- or
           set @r = 1
        else  --test
           set @r = 3
";
			Parse(sql);

			ThenExprShouldBe(@"IF @id IN (1, 2)
SET @r = 0
ELSE IF @id = 2
SET @r = 1
ELSE
SET @r = 3
");
		}

		[Fact]
		public void if_not_in()
		{
			var sql = @"if @id not in (1,2)
begin
	select 1
end
";
			Parse(sql);

			ThenExprShouldBe(@"IF @id NOT IN (1, 2)
BEGIN
	SELECT 1
END");
		}


		[Fact]
		public void if_else_begin_tran()
		{
			var sql = @"if @id = 1
begin
	select 1
end else begin tran
";
			Parse(sql);

			ThenExprShouldBe(@"IF @id = 1
BEGIN 
	SELECT 1
END ELSE BEGIN TRANSACTION");
		}
	}
}
