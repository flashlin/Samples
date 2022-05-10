using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class SetTest : TestBase
	{
		public SetTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void set_ANSI_NULLS()
		{
			var sql = "set ansi_nulls off";
			Parse(sql);
			ThenExprShouldBe("SET ANSI_NULLS OFF");
		}

		[Fact]
		public void script_servar_var_doubleQuoteString()
		{
			var sql = ":setvar id \"123\"";
			Parse(sql);
			ThenExprShouldBe(":SETVAR id \"123\"");
		}

		[Fact]
		public void set_identity_insert_table_off()
		{
			var sql = "set identity_insert customer off";
			Parse(sql);
			ThenExprShouldBe("SET IDENTITY_INSERT customer OFF");
		}

		[Fact]
		public void set_var_eq()
		{
			var sql = "set @id = 1";
			Parse(sql);
			ThenExprShouldBe("SET @id = 1");
		}

		[Fact]
		public void set_var_eq_customFunc()
		{
			var sql = "set @isFlag = [dbo].[fn_my]()";
			Parse(sql);
			ThenExprShouldBe("SET @isFlag = [dbo].[fn_my]()");
		}

		[Fact]
		public void set_name_eq_customFunc()
		{
			var sql = "set name = [dbo].[fn_my](a,b)";
			Parse(sql);
			ThenExprShouldBe("SET name = [dbo].[fn_my]( a, b )");
		}

		[Fact]
		public void dateadd()
		{
			var sql = "dateadd(day, datediff(day,1,GETDATE()), 0)";
			Parse(sql);
			ThenExprShouldBe("DATEADD( day, DATEDIFF( day, 1, GETDATE() ), 0 )");
		}
	}
}
