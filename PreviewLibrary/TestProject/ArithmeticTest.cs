using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.RecursiveParser;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ArithmeticTest : SqlTestBase
	{
		public ArithmeticTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void add_mul()
		{
			var sql = "1 + 2 * 3";
			var expr = new SqlParser().ParseArithmeticPartial(sql);
			new OperandExpr
			{
				Left = new IntegerExpr
				{
					Value = 1
				},
				Oper = "+",
				Right = new OperandExpr
				{
					Left = new IntegerExpr
					{
						Value = 2
					},
					Oper = "*",
					Right = new IntegerExpr
					{
						Value = 3
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void add_mul_add()
		{
			var sql = "1 + 2 * 3 + 4";
			var expr = new SqlParser().ParseArithmeticPartial(sql);
			"1 + 2 * 3 + 4".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void mul_first_add()
		{
			var sql = "1 * ( 2 + 3)";
			var expr = _sqlParser.ParseArithmeticPartial(sql);
			"1 * (2 + 3)".ShouldEqual(expr);
		}

		[Fact]
		public void a_and_b()
		{
			var sql = "a & @b";
			var expr = _sqlParser.ParseArithmeticPartial(sql);
			"a & @b".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void multiple_group()
		{
			var sql = "((a + 1) * b)";
			var expr = _sqlParser.ParseArithmeticPartial(sql);
			"((a + 1) * b)".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_sub_b()
		{
			var sql = "a -1";
			var expr = _sqlParser.ParseArithmeticPartial(sql);
			"a - 1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_or_b()
		{
			var sql = "@a | @b";
			var expr = _sqlParser.ParseArithmeticPartial(sql);
			"@a | @b".ShouldEqual(expr);
		}


		[Fact]
		public void nest()
		{
			var sql = @"
   (
		(@a - @b) + (@c - @d) + 
		(@b1 -b2) + (b3 - b4)
	)";
			var expr = _sqlParser.ParseArithmeticPartial(sql);

			"((@a - @b) + (@c - @d) + (@b1 - b2) + (b3 - b4))".ShouldEqual(expr);
		}
	}


	public class FilterTest : SqlTestBase
	{
		public FilterTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void expr_notEqual_expr()
		{
			var sql = "1 <> 2";
			var expr = _sqlParser.ParseFilterPartial(sql);
			"1 <> 2".ShouldEqual(expr);
		}
		
		[Fact]
		public void func_notEqual_expr()
		{
			var sql = "isnull(@name, '') <> ''";
			var expr = _sqlParser.ParseFilterPartial(sql);
			"isnull( @name,'' ) <> ''".ShouldEqual(expr);
		}
	}
}