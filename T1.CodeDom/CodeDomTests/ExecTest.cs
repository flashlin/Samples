﻿using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class ExecTest : TestBase
	{
		public ExecTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void exec_func_a_b()
		{
			var sql = @"exec myFunc a,b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc a, b ");
		}

		[Fact]
		public void exec_var_eq()
		{
			var sql = @"exec @a = myfunc a,b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC @a = myfunc a, b");
		}

		[Fact]
		public void exec_func_var_out()
		{
			var sql = @"exec myFunc @a out,b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc @a OUT, b");
		}


		[Fact]
		public void exec_func_a_var_eq_b()
		{
			var sql = @"exec myFunc a, @b=b";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc a, @b = b");
		}

		[Fact]
		public void exec_func_var_eq_b()
		{
			var sql = @"exec myFunc @a='b'";
			Parse(sql);

			ThenExprShouldBe(@"EXEC myFunc @a = 'b'");
		}

	}
}
