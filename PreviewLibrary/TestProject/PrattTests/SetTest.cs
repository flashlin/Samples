using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
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


	}
}
