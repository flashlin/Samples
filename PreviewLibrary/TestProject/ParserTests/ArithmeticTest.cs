using FluentAssertions;
using System;
using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class ArithmeticTest : ParserTestBase
	{
		public ArithmeticTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void add()
		{
			var sql = "1 + 2";
			Parse(sql);
			ThenExprShouldBe("1 + 2");
		}

		protected void ThenExprShouldBe(string expect)
		{
			_expr.Token.Should().Be(expect);
		}
	}
}
