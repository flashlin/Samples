using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
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
		public void case_when()
		{
			var sql = "CASE WHEN @a = -1 THEN [b] ELSE @c END";
			var expr = _sqlParser.ParseCasePartial(sql);
			@"CASE
	WHEN @a = -1 THEN [b]
	ELSE @c
END".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void table()
		{
			var sql = @"table
(
	BetOption nvarchar(10)
)";
			var expr = _sqlParser.ParseDataTypePartial(sql);
			new TableTypeExpr
			{
				ColumnTypeList = new List<SqlExpr>
				{
					new DefineColumnTypeExpr
					{
						 Name = new IdentExpr
						 {
							  Name = "BetOption"
						 },
						 DataType = new DataTypeExpr
						 {
							  DataType = "nvarchar",
							  DataSize = new DataTypeSizeExpr
							  {
									Size = 10
							  }
						 }
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
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
		public void if_notEqual_expr()
		{
			var sql = @"if( isnull(@name, '') <> '' )
BEGIN select 1 END";
			var expr = _sqlParser.ParseIfPartial(sql);
			@"IF (isnull( @name,'' ) <> '')
BEGIN
SELECT 1
END".ToExpectedObject().ShouldEqual(expr.ToString());
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
		public void isnull()
		{
			var sql = "isnull(@betCondition, '')";
			var expr = _sqlParser.ParseFuncPartial(sql);
			new SqlFuncExpr
			{
				Name = "isnull",
				Arguments = new SqlExpr[]
				{
					new IdentExpr
					{
						Name = "@betCondition"
					},new StringExpr
					{
						Text = "''"
					}
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