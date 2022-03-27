using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class SqlPartialExprTest : SqlTestBase
	{
		public SqlPartialExprTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void expr_notEqual_expr()
		{
			var sql = "1 <> 2";
			var expr = _sqlParser.ParseFilterPartial(sql);
			new CompareExpr
			{
				Left = new IntegerExpr
				{
					Value = 1
				},
				Oper = "<>",
				Right = new IntegerExpr
				{
					Value = 2
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void func_notEqual_expr()
		{
			var sql = "isnull(@name, '') <> ''";
			var expr = _sqlParser.ParseFilterPartial(sql);
			new CompareExpr
			{
				Left = new SqlFuncExpr
				{
					Name = "isnull",
					Arguments = new SqlExpr[]
					{
						new IdentExpr
						{
						  Name = "@name"
						},
						new StringExpr
						{
						  Text = "''"
						}
					}
				},
				Oper = "<>",
				Right = new StringExpr
				{
					Text = "''"
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void unknown_custom_func()
		{
			var sql = "strsplitmax(@a, N',')";
			var expr = _sqlParser.ParseFuncPartial(sql);
			new CustomFuncExpr
			{
				ObjectId = new IdentExpr
				{
					Name = "strsplitmax"
				},
				Name = "strsplitmax",
				Arguments = new SqlExpr[] { 
					new IdentExpr
					{
						Name = "@a"
					},new StringExpr
					{
						Text = "N','"
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}