﻿using System;
using FluentAssertions;
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

		[Fact]
		public void mul()
		{
			var sql = "1 * 2";
			Parse(sql);
			ThenExprShouldBe("1 * 2");
		}

		[Fact]
		public void divide()
		{
			var sql = "1 / 2";
			Parse(sql);
			ThenExprShouldBe("1 / 2");
		}

		[Fact]
		public void paren_add()
		{
			var sql = "(1 + 2)";
			Parse(sql);
			ThenExprShouldBe("( 1 + 2 )");
		}
	}
}
